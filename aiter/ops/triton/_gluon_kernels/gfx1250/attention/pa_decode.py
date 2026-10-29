# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon gfx1250 paged attention decode kernel.

The wrapper ``aiter/ops/triton/gluon/pa_decode_gluon.py`` launches this kernel
on gfx1250. The kernel uses WMMA and TDM async loads with two or three LDS buffers.

Supports:
  - BF16/FP16 compute, FP8 compute, and BF16 compute with an FP8 KV cache
  - Per-tensor and per-token Q and KV scales
  - query_length 1-4 with causal masking, sliding window and attention sinks
  - KV_BLOCK_SIZE 16-1024, Q/K and V head sizes up to 256 (padded to pow2)
  - 5D K cache [num_blocks, num_kv_heads, head_size // x, kv_block_size, x]
  - 4D V cache [num_blocks, num_kv_heads, value_head_size, kv_block_size] or
    5D V cache [num_blocks, num_kv_heads, kv_block_size // x, value_head_size, x]
  - One-shot output, or split partials for the PS reduce kernel
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_pa_decode_gfx1250_repr = make_kernel_repr(
    "_pa_decode_gluon_gfx1250",
    [
        "BLOCK_M",
        "HEAD_SIZE_POW2",
        "VALUE_HEAD_SIZE_POW2",
        "KV_BLOCK_SIZE",
        "TILE_SIZE",
        "ONE_SHOT",
    ],
)


@gluon.jit
def _issue_tile_load(
    tile_idx,
    buf,
    k_desc,
    v_desc,
    k_smem,
    v_smem,
    block_tables_base,
    last_page,
    num_kv_heads,
    kv_head_idx,
    KV_BLOCK_SIZE: gl.constexpr,
    TILE_SIZE: gl.constexpr,
    K_X: gl.constexpr,
    V_X: gl.constexpr,
    VALUE_TRANSPOSED: gl.constexpr,
    KV_GATHER: gl.constexpr,
):
    if KV_GATHER:
        # The tile is several whole pages: one TDM gather of page rows each.
        PAGES: gl.constexpr = TILE_SIZE // KV_BLOCK_SIZE
        ids_layout: gl.constexpr = gl.BlockedLayout([PAGES], [32], [4], [0])
        slots = tile_idx * PAGES + gl.arange(0, PAGES, layout=ids_layout)
        # Clamp to the last valid page so that padded tokens hold finite data.
        pages = gl.load(block_tables_base + gl.minimum(slots, last_page))
        rows = (pages * num_kv_heads + kv_head_idx).to(gl.int32)
        gl.amd.gfx1250.tdm.async_gather(k_desc, rows, k_smem.index(buf))
        gl.amd.gfx1250.tdm.async_gather(v_desc, rows, v_smem.index(buf))
    else:
        # The tile is inside one page. Tiles past the context are never issued.
        token = tile_idx * TILE_SIZE
        page = gl.load(block_tables_base + token // KV_BLOCK_SIZE)
        page_off = token % KV_BLOCK_SIZE
        row = (page * num_kv_heads + kv_head_idx).to(gl.int32)
        gl.amd.gfx1250.tdm.async_load(
            k_desc, [row, 0, (page_off * K_X).to(gl.int32)], k_smem.index(buf)
        )
        if VALUE_TRANSPOSED:
            gl.amd.gfx1250.tdm.async_load(
                v_desc,
                [row, (page_off // V_X).to(gl.int32), 0, 0],
                v_smem.index(buf),
            )
        else:
            gl.amd.gfx1250.tdm.async_load(
                v_desc, [row, 0, page_off.to(gl.int32)], v_smem.index(buf)
            )


@gluon.jit
def _gather_kv_tile(
    tile_start,
    key_cache_ptr,
    value_cache_ptr,
    block_tables_base,
    last_page,
    context_length,
    num_kv_heads,
    kv_head_idx,
    stride_key_head,
    stride_value_head,
    head_size,
    value_head_size,
    HEAD_SIZE_POW2: gl.constexpr,
    VALUE_HEAD_SIZE_POW2: gl.constexpr,
    KV_BLOCK_SIZE: gl.constexpr,
    TILE_SIZE: gl.constexpr,
    K_X: gl.constexpr,
    V_X: gl.constexpr,
    VALUE_TRANSPOSED: gl.constexpr,
):
    # Tiles that span several small pages: gather K/V with vector loads.
    # Masked slots read zero, so padded tokens and head dims are finite.
    K_LOAD: gl.constexpr = gl.BlockedLayout([K_X, 1], [1, 32], [1, 4], [0, 1])
    k_d = gl.arange(0, HEAD_SIZE_POW2, layout=gl.SliceLayout(1, K_LOAD))
    k_t = tile_start + gl.arange(0, TILE_SIZE, layout=gl.SliceLayout(0, K_LOAD))
    k_page = gl.load(block_tables_base + gl.minimum(k_t // KV_BLOCK_SIZE, last_page))
    k_row = (k_page.to(gl.int64) * num_kv_heads + kv_head_idx) * stride_key_head
    k_offs = (
        k_row[None, :]
        + (k_d // K_X)[:, None] * (KV_BLOCK_SIZE * K_X)
        + (k_t % KV_BLOCK_SIZE)[None, :] * K_X
        + (k_d % K_X)[:, None]
    )
    k_mask = (k_d < head_size)[:, None] & (k_t < context_length)[None, :]
    k_tile = gl.load(key_cache_ptr + k_offs, mask=k_mask, other=0.0)

    V_VEC: gl.constexpr = V_X if VALUE_TRANSPOSED else 8
    V_LOAD: gl.constexpr = gl.BlockedLayout([V_VEC, 1], [1, 32], [1, 4], [0, 1])
    v_t = tile_start + gl.arange(0, TILE_SIZE, layout=gl.SliceLayout(1, V_LOAD))
    v_d = gl.arange(0, VALUE_HEAD_SIZE_POW2, layout=gl.SliceLayout(0, V_LOAD))
    v_page = gl.load(block_tables_base + gl.minimum(v_t // KV_BLOCK_SIZE, last_page))
    v_row = (v_page.to(gl.int64) * num_kv_heads + kv_head_idx) * stride_value_head
    v_in_page = v_t % KV_BLOCK_SIZE
    if VALUE_TRANSPOSED:
        v_offs = (
            v_row[:, None]
            + (v_in_page // V_X)[:, None] * (value_head_size * V_X)
            + v_d[None, :] * V_X
            + (v_in_page % V_X)[:, None]
        )
    else:
        v_offs = v_row[:, None] + v_d[None, :] * KV_BLOCK_SIZE + v_in_page[:, None]
    v_mask = (v_t < context_length)[:, None] & (v_d < value_head_size)[None, :]
    v_tile = gl.load(value_cache_ptr + v_offs, mask=v_mask, other=0.0)
    return k_tile, v_tile


@gluon.jit
def _load_kv_scales(
    tile_idx,
    block_tables_base,
    last_page,
    context_length,
    key_scale,
    value_scale,
    kv_scale_stride_0,
    kv_scale_stride_1,
    kv_head_idx,
    KV_QUANT_MODE: gl.constexpr,
    KV_BLOCK_SIZE: gl.constexpr,
    TILE_SIZE: gl.constexpr,
    QK_LAYOUT: gl.constexpr,
):
    # Per-token K/V scales of one tile. Other quant modes return dummies.
    if KV_QUANT_MODE == 1:
        # Coalesced 1D loads, then one layout conversion to the score columns.
        SCALE_LAYOUT: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
        tile_start = tile_idx * TILE_SIZE
        cols = tile_start + gl.arange(0, TILE_SIZE, layout=SCALE_LAYOUT)
        col_valid = cols < context_length
        if TILE_SIZE <= KV_BLOCK_SIZE:
            # The tile is inside one page: one block-table read.
            pages = gl.load(block_tables_base + tile_start // KV_BLOCK_SIZE).to(
                gl.int64
            )
        else:
            page_slot = gl.minimum(cols // KV_BLOCK_SIZE, last_page)
            pages = gl.load(block_tables_base + page_slot).to(gl.int64)
        scale_offs = (
            pages * kv_scale_stride_0
            + kv_head_idx * kv_scale_stride_1
            + cols % KV_BLOCK_SIZE
        )
        k_scales = gl.load(key_scale + scale_offs, mask=col_valid, other=0.0)
        v_scales = gl.load(value_scale + scale_offs, mask=col_valid, other=0.0)
        COL_LAYOUT: gl.constexpr = gl.SliceLayout(0, QK_LAYOUT)
        return (
            gl.convert_layout(k_scales.to(gl.float32), COL_LAYOUT),
            gl.convert_layout(v_scales.to(gl.float32), COL_LAYOUT),
        )
    else:
        return 0.0, 0.0


@gluon.jit
def _process_tile(
    tile_idx,
    buf,
    k_regs,
    v_regs,
    tile_k_scale,
    tile_v_scale,
    q_dot,
    m_i,
    l_i,
    acc,
    k_smem,
    v_smem,
    block_tables_base,
    last_page,
    context_length,
    window_lo,
    qk_scale,
    row_valid,
    row_pos,
    per_token_q_scale,
    key_scale_value,
    value_scale_value,
    kv_head_idx,
    query_seq_len,
    COMPUTE_TYPE: gl.constexpr,
    BLOCK_M: gl.constexpr,
    HEAD_SIZE_POW2: gl.constexpr,
    VALUE_HEAD_SIZE_POW2: gl.constexpr,
    KV_BLOCK_SIZE: gl.constexpr,
    TILE_SIZE: gl.constexpr,
    K_X: gl.constexpr,
    V_X: gl.constexpr,
    VALUE_TRANSPOSED: gl.constexpr,
    QUERY_QUANT_MODE: gl.constexpr,
    KV_QUANT_MODE: gl.constexpr,
    IS_FP8: gl.constexpr,
    FP8_MAX_VALUE: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
    SLIDING_WINDOW: gl.constexpr,
    MASK_V: gl.constexpr,
    USE_TDM: gl.constexpr,
    KV_GATHER: gl.constexpr,
    QK_LAYOUT: gl.constexpr,
    PV_LAYOUT: gl.constexpr,
    DOT_K_LAYOUT: gl.constexpr,
    DOT_P_LAYOUT: gl.constexpr,
    DOT_V_LAYOUT: gl.constexpr,
):
    PAGES: gl.constexpr = TILE_SIZE // KV_BLOCK_SIZE if KV_GATHER else 1
    PAGE_TOKENS: gl.constexpr = TILE_SIZE // PAGES
    tile_start = tile_idx * TILE_SIZE
    cols = tile_start + gl.arange(0, TILE_SIZE, layout=gl.SliceLayout(0, QK_LAYOUT))
    col_valid = cols < context_length
    if USE_TDM:
        # LDS K: [P, D/x, PAGE_TOKENS * x] -> [TILE, D] -> [D, TILE]
        k_tile = (
            k_smem.index(buf)
            .reshape([PAGES, HEAD_SIZE_POW2 // K_X, PAGE_TOKENS, K_X])
            .permute([0, 2, 1, 3])
            .reshape([TILE_SIZE, HEAD_SIZE_POW2])
            .permute([1, 0])
            .load(DOT_K_LAYOUT)
        )
    else:
        k_tile = k_regs
    k_tile = k_tile.to(COMPUTE_TYPE)
    scores = gl.amd.gfx1250.wmma(
        q_dot,
        k_tile,
        gl.zeros([BLOCK_M, TILE_SIZE], dtype=gl.float32, layout=QK_LAYOUT),
    )

    scores = scores * qk_scale
    if KV_QUANT_MODE == 1:
        scores = scores * tile_k_scale[None, :]
    if QUERY_QUANT_MODE == 1:
        scores = scores * per_token_q_scale[:, None]

    # Query row t sees keys [ctx - SW + t + 1, ctx - qlen + t].
    mask = row_valid[:, None] & col_valid[None, :]
    if IS_CAUSAL:
        causal_ext = query_seq_len - 1 - row_pos
        mask = mask & ((causal_ext[:, None] + cols[None, :]) < context_length)
    if SLIDING_WINDOW > 0:
        mask = mask & (cols[None, :] >= window_lo + row_pos[:, None])
    scores = gl.where(mask, scores, float("-inf"))

    # Online softmax in the log2 domain.
    m_new = gl.maximum(m_i, gl.max(scores, axis=1))
    m_safe = gl.where(m_new == float("-inf"), 0.0, m_new)
    alpha = gl.exp2(m_i - m_safe)
    p = gl.exp2(scores - m_safe[:, None])
    l_i = l_i * alpha + gl.sum(p, axis=1)

    probability_scale = 1.0
    if KV_QUANT_MODE == 0:
        if IS_FP8:
            p = p * float(FP8_MAX_VALUE)
            probability_scale = value_scale_value / float(FP8_MAX_VALUE)
        else:
            probability_scale = value_scale_value
    elif KV_QUANT_MODE == 1:
        if IS_FP8:
            v_scale_max = gl.max(tile_v_scale, axis=0)
            p = (
                p
                * (tile_v_scale * (float(FP8_MAX_VALUE) / (v_scale_max + 1e-8)))[
                    None, :
                ]
            )
            probability_scale = v_scale_max / float(FP8_MAX_VALUE)
        else:
            p = p * tile_v_scale[None, :]

    if not USE_TDM:
        v_tile = v_regs
    else:
        if VALUE_TRANSPOSED:
            # LDS V: [TILE / x, Dv, x] -> [TILE, Dv]
            v_view = (
                v_smem.index(buf)
                .reshape([TILE_SIZE // V_X, VALUE_HEAD_SIZE_POW2, V_X])
                .permute([0, 2, 1])
                .reshape([TILE_SIZE, VALUE_HEAD_SIZE_POW2])
            )
        else:
            # LDS V: [P, Dv, PAGE_TOKENS] -> [TILE, Dv]
            v_view = (
                v_smem.index(buf)
                .reshape([PAGES, VALUE_HEAD_SIZE_POW2, PAGE_TOKENS])
                .permute([1, 0, 2])
                .reshape([VALUE_HEAD_SIZE_POW2, TILE_SIZE])
                .permute([1, 0])
            )
        v_tile = v_view.load(DOT_V_LAYOUT)
        if MASK_V:
            # Slots past the context can hold NaN, and 0 * NaN is NaN.
            v_rows = tile_start + gl.arange(
                0, TILE_SIZE, layout=gl.SliceLayout(1, DOT_V_LAYOUT)
            )
            v_tile = gl.where((v_rows < context_length)[:, None], v_tile, 0.0)
    v_tile = v_tile.to(COMPUTE_TYPE)

    p_dot = gl.convert_layout(p.to(COMPUTE_TYPE), DOT_P_LAYOUT)
    alpha_pv = gl.convert_layout(alpha, gl.SliceLayout(1, PV_LAYOUT))
    pv = gl.amd.gfx1250.wmma(
        p_dot,
        v_tile,
        gl.zeros([BLOCK_M, VALUE_HEAD_SIZE_POW2], dtype=gl.float32, layout=PV_LAYOUT),
    )
    if KV_QUANT_MODE >= 0:
        acc = acc * alpha_pv[:, None] + probability_scale * pv
    else:
        acc = acc * alpha_pv[:, None] + pv
    return m_new, l_i, acc


@gluon.jit(repr=_pa_decode_gfx1250_repr)
def _pa_decode_gluon_gfx1250(
    exp_sums_ptr,  # [num_seqs, num_kv_heads, max_parts, query_length * group_size]
    max_logits_ptr,  # [num_seqs, num_kv_heads, max_parts, query_length * group_size]
    output_ptr,  # one-shot: [bs, qlen, kv_heads, group, Dv]; else [bs, kv_heads, parts, qlen * group, Dv]
    query_ptr,  # [bs, qlen, kv_heads, group, D]
    key_cache_ptr,
    value_cache_ptr,
    block_tables_ptr,
    context_lengths_ptr,
    softmax_scale,
    query_scale,
    key_scale,
    value_scale,
    sinks_ptr,
    stride_max_logits_seq,
    stride_max_logits_head,
    stride_max_logits_part,
    stride_output_0,
    stride_output_1,
    stride_output_2,
    stride_output_3,
    stride_query_bs,
    stride_query_qlen,
    stride_query_kv_head,
    stride_query_group_size,
    stride_key_head,
    stride_value_head,
    stride_block_table_seq,
    stride_query_scale_bs,
    stride_query_scale_qlen,
    stride_query_scale_kv_head,
    kv_scale_stride_0,
    kv_scale_stride_1,
    num_blocks,
    num_kv_heads,
    query_seq_len,
    query_group_size,
    head_size,
    value_head_size,
    COMPUTE_TYPE: gl.constexpr,
    QUERY_SEQ_LEN_POW2: gl.constexpr,
    ONE_QUERY_GROUP_SIZE_POW2: gl.constexpr,
    HEAD_SIZE: gl.constexpr,
    HEAD_SIZE_POW2: gl.constexpr,
    VALUE_HEAD_SIZE: gl.constexpr,
    VALUE_HEAD_SIZE_POW2: gl.constexpr,
    KV_BLOCK_SIZE: gl.constexpr,
    K_X: gl.constexpr,
    V_X: gl.constexpr,
    QUERY_QUANT_MODE: gl.constexpr,
    KV_QUANT_MODE: gl.constexpr,
    VALUE_TRANSPOSED: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
    FP8_MAX_VALUE: gl.constexpr,
    SLIDING_WINDOW: gl.constexpr,
    ONE_SHOT: gl.constexpr,
    IS_HEAD1: gl.constexpr,
    BLOCK_M: gl.constexpr,
    TILE_SIZE: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
):
    NUM_WARPS: gl.constexpr = 4
    WARP_SIZE: gl.constexpr = 32
    LOG2E: gl.constexpr = 1.4426950408889634
    LN2: gl.constexpr = 0.6931471805599453
    IS_FP8: gl.constexpr = COMPUTE_TYPE.is_fp8()
    # TDM loads a page slice per tile, or gathers whole small pages. Small
    # pages with padded head sizes use a register gather.
    KV_GATHER: gl.constexpr = TILE_SIZE > KV_BLOCK_SIZE
    USE_TDM: gl.constexpr = not KV_GATHER or (
        HEAD_SIZE_POW2 == HEAD_SIZE and VALUE_HEAD_SIZE_POW2 == VALUE_HEAD_SIZE
    )
    PAGES: gl.constexpr = TILE_SIZE // KV_BLOCK_SIZE if KV_GATHER else 1
    OUTPUT_DTYPE: gl.constexpr = output_ptr.dtype.element_ty

    # FP8 WMMA is [16, 16, 64/128] with k_width 16; 16-bit WMMA is [16, 16, 32].
    if IS_FP8:
        WMMA_K_QK: gl.constexpr = 128 if HEAD_SIZE_POW2 > 64 else 64
        WMMA_K_PV: gl.constexpr = 128 if TILE_SIZE > 64 else 64
        K_WIDTH: gl.constexpr = 16
    else:
        WMMA_K_QK: gl.constexpr = 32
        WMMA_K_PV: gl.constexpr = 32
        K_WIDTH: gl.constexpr = 8

    # With 64+ rows, warps split rows and the softmax stays inside a warp
    # (FP8 is slower so). Else warps split N: tokens for QK, value head for PV.
    if BLOCK_M >= 16 * NUM_WARPS and not IS_FP8:
        WARP_BASES: gl.constexpr = [[1, 0], [2, 0]]
    else:
        WARP_BASES: gl.constexpr = [[0, 1], [0, 2]]
    QK_LAYOUT: gl.constexpr = gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        instr_shape=[16, 16, WMMA_K_QK],
        warp_bases=WARP_BASES,
    )
    PV_LAYOUT: gl.constexpr = gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        instr_shape=[16, 16, WMMA_K_PV],
        warp_bases=WARP_BASES,
    )
    DOT_Q_LAYOUT: gl.constexpr = gl.DotOperandLayout(0, QK_LAYOUT, K_WIDTH)
    DOT_K_LAYOUT: gl.constexpr = gl.DotOperandLayout(1, QK_LAYOUT, K_WIDTH)
    DOT_P_LAYOUT: gl.constexpr = gl.DotOperandLayout(0, PV_LAYOUT, K_WIDTH)
    DOT_V_LAYOUT: gl.constexpr = gl.DotOperandLayout(1, PV_LAYOUT, K_WIDTH)
    Q_LANES_D: gl.constexpr = min(WARP_SIZE, HEAD_SIZE_POW2 // 8)
    Q_LAYOUT: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[WARP_SIZE // Q_LANES_D, Q_LANES_D],
        warps_per_cta=[NUM_WARPS, 1],
        order=[1, 0],
    )
    ROW_QK: gl.constexpr = gl.SliceLayout(1, QK_LAYOUT)
    ROW_PV: gl.constexpr = gl.SliceLayout(1, PV_LAYOUT)

    K_SMEM_LAYOUT: gl.constexpr = gl.SwizzledSharedLayout(
        vec=1, per_phase=1, max_phase=1, order=[2, 1, 0]
    )
    V_SMEM_LAYOUT: gl.constexpr = gl.SwizzledSharedLayout(
        vec=1,
        per_phase=1,
        max_phase=1,
        order=[3, 2, 1, 0] if VALUE_TRANSPOSED else [2, 1, 0],
    )

    # ==================== Program IDs ====================
    sequence_idx = gl.program_id(0)
    if IS_HEAD1:
        # One KV head: program_id(1) selects a group of query positions.
        kv_head_idx = 0
        query_begin = gl.program_id(1) * QUERY_SEQ_LEN_POW2
    else:
        kv_head_idx = gl.program_id(1)
        query_begin = 0
    split_idx = gl.program_id(2)
    num_splits = gl.num_programs(2)
    query_end = gl.minimum(query_seq_len, query_begin + QUERY_SEQ_LEN_POW2)

    context_length = gl.load(context_lengths_ptr + sequence_idx)

    # ==================== Tiles of this split ====================
    if SLIDING_WINDOW > 0:
        window_lo = context_length - SLIDING_WINDOW + 1
    else:
        window_lo = context_length * 0
    tile_lo = gl.maximum(window_lo, 0) // TILE_SIZE
    tile_hi = gl.cdiv(context_length, TILE_SIZE)
    tiles_per_split = gl.cdiv(tile_hi - tile_lo, num_splits)
    split_tile_start = tile_lo + split_idx * tiles_per_split
    split_tile_end = gl.minimum(tile_hi, split_tile_start + tiles_per_split)
    num_tiles = split_tile_end - split_tile_start

    # ==================== Rows: (query position, group head) ====================
    rows = gl.arange(0, BLOCK_M, layout=ROW_QK)
    row_pos = query_begin + rows // ONE_QUERY_GROUP_SIZE_POW2
    row_head = rows % ONE_QUERY_GROUP_SIZE_POW2
    row_valid = (row_pos < query_end) & (row_head < query_group_size)
    out_row = row_pos * query_group_size + row_head

    # ==================== Q ====================
    q_rows = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, Q_LAYOUT))
    q_cols = gl.arange(0, HEAD_SIZE_POW2, layout=gl.SliceLayout(0, Q_LAYOUT))
    q_pos = query_begin + q_rows // ONE_QUERY_GROUP_SIZE_POW2
    q_head = q_rows % ONE_QUERY_GROUP_SIZE_POW2
    q_row_valid = (q_pos < query_end) & (q_head < query_group_size)
    q_offsets = (
        sequence_idx.to(gl.int64) * stride_query_bs
        + kv_head_idx * stride_query_kv_head
        + q_pos[:, None] * stride_query_qlen
        + q_head[:, None] * stride_query_group_size
        + q_cols[None, :]
    )
    q = gl.load(
        query_ptr + q_offsets,
        mask=q_row_valid[:, None] & (q_cols[None, :] < head_size),
        other=0.0,
    )
    q_dot = gl.convert_layout(q.to(COMPUTE_TYPE), DOT_Q_LAYOUT)

    qk_scale = softmax_scale * LOG2E
    if QUERY_QUANT_MODE == 0:
        qk_scale = qk_scale * gl.load(query_scale).to(gl.float32)
    key_scale_value = 1.0
    value_scale_value = 1.0
    if KV_QUANT_MODE == 0:
        key_scale_value = gl.load(key_scale).to(gl.float32)
        value_scale_value = gl.load(value_scale).to(gl.float32)
        qk_scale = qk_scale * key_scale_value
    per_token_q_scale = gl.full([BLOCK_M], 1.0, dtype=gl.float32, layout=ROW_QK)
    if QUERY_QUANT_MODE == 1:
        q_scale_offsets = (
            sequence_idx * stride_query_scale_bs
            + row_pos * stride_query_scale_qlen
            + kv_head_idx * stride_query_scale_kv_head
            + row_head
        )
        per_token_q_scale = gl.load(
            query_scale + q_scale_offsets, mask=row_valid, other=1.0
        ).to(gl.float32)

    m_i = gl.full([BLOCK_M], float("-inf"), dtype=gl.float32, layout=ROW_QK)
    l_i = gl.full([BLOCK_M], 0.0, dtype=gl.float32, layout=ROW_QK)
    acc = gl.zeros([BLOCK_M, VALUE_HEAD_SIZE_POW2], dtype=gl.float32, layout=PV_LAYOUT)

    block_tables_base = (
        block_tables_ptr + sequence_idx.to(gl.int64) * stride_block_table_seq
    )
    last_page = gl.maximum(gl.cdiv(context_length, KV_BLOCK_SIZE) - 1, 0)

    if num_tiles > 0 and USE_TDM:
        # ==================== TDM descriptors ====================
        GATHER_LAYOUT: gl.constexpr = gl.SwizzledSharedLayout(
            vec=1, per_phase=1, max_phase=1, order=[1, 0]
        )
        if KV_GATHER:
            # One row per (page, kv head): [blocks * kv_heads, page * head size]
            k_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
                base=key_cache_ptr,
                shape=(num_blocks * num_kv_heads, KV_BLOCK_SIZE * HEAD_SIZE),
                strides=(stride_key_head, 1),
                block_shape=(PAGES, KV_BLOCK_SIZE * HEAD_SIZE),
                layout=GATHER_LAYOUT,
            )
            k_smem = gl.allocate_shared_memory(
                key_cache_ptr.dtype.element_ty,
                [NUM_BUFFERS, PAGES, KV_BLOCK_SIZE * HEAD_SIZE],
                GATHER_LAYOUT,
            )
            v_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
                base=value_cache_ptr,
                shape=(num_blocks * num_kv_heads, KV_BLOCK_SIZE * VALUE_HEAD_SIZE),
                strides=(stride_value_head, 1),
                block_shape=(PAGES, KV_BLOCK_SIZE * VALUE_HEAD_SIZE),
                layout=GATHER_LAYOUT,
            )
            v_smem = gl.allocate_shared_memory(
                value_cache_ptr.dtype.element_ty,
                [NUM_BUFFERS, PAGES, KV_BLOCK_SIZE * VALUE_HEAD_SIZE],
                GATHER_LAYOUT,
            )
        else:
            # K: [blocks * kv_heads, D / x, KV_BLOCK_SIZE * x]; padded D reads zeros.
            k_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
                base=key_cache_ptr,
                shape=(
                    num_blocks * num_kv_heads,
                    head_size // K_X,
                    KV_BLOCK_SIZE * K_X,
                ),
                strides=(stride_key_head, KV_BLOCK_SIZE * K_X, 1),
                block_shape=(1, HEAD_SIZE_POW2 // K_X, TILE_SIZE * K_X),
                layout=K_SMEM_LAYOUT,
            )
            k_smem = gl.allocate_shared_memory(
                key_cache_ptr.dtype.element_ty,
                [NUM_BUFFERS, 1, HEAD_SIZE_POW2 // K_X, TILE_SIZE * K_X],
                K_SMEM_LAYOUT,
            )
            if VALUE_TRANSPOSED:
                # V: [blocks * kv_heads, KV_BLOCK_SIZE / x, Dv, x]
                v_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
                    base=value_cache_ptr,
                    shape=(
                        num_blocks * num_kv_heads,
                        KV_BLOCK_SIZE // V_X,
                        value_head_size,
                        V_X,
                    ),
                    strides=(stride_value_head, value_head_size * V_X, V_X, 1),
                    block_shape=(1, TILE_SIZE // V_X, VALUE_HEAD_SIZE_POW2, V_X),
                    layout=V_SMEM_LAYOUT,
                )
                v_smem = gl.allocate_shared_memory(
                    value_cache_ptr.dtype.element_ty,
                    [NUM_BUFFERS, 1, TILE_SIZE // V_X, VALUE_HEAD_SIZE_POW2, V_X],
                    V_SMEM_LAYOUT,
                )
            else:
                # V: [blocks * kv_heads, Dv, KV_BLOCK_SIZE]
                v_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
                    base=value_cache_ptr,
                    shape=(num_blocks * num_kv_heads, value_head_size, KV_BLOCK_SIZE),
                    strides=(stride_value_head, KV_BLOCK_SIZE, 1),
                    block_shape=(1, VALUE_HEAD_SIZE_POW2, TILE_SIZE),
                    layout=V_SMEM_LAYOUT,
                )
                v_smem = gl.allocate_shared_memory(
                    value_cache_ptr.dtype.element_ty,
                    [NUM_BUFFERS, 1, VALUE_HEAD_SIZE_POW2, TILE_SIZE],
                    V_SMEM_LAYOUT,
                )

        # ==================== Prologue: NUM_BUFFERS tiles in flight ====================
        for j in gl.static_range(NUM_BUFFERS):
            if j < num_tiles:
                _issue_tile_load(
                    split_tile_start + j, j, k_desc, v_desc, k_smem, v_smem,
                    block_tables_base, last_page, num_kv_heads, kv_head_idx,
                    KV_BLOCK_SIZE, TILE_SIZE, K_X, V_X, VALUE_TRANSPOSED, KV_GATHER,
                )  # fmt: skip
        k_scales, v_scales = _load_kv_scales(
            split_tile_start, block_tables_base, last_page, context_length, key_scale, value_scale,
                kv_scale_stride_0, kv_scale_stride_1, kv_head_idx, KV_QUANT_MODE,
                KV_BLOCK_SIZE, TILE_SIZE, QK_LAYOUT,
        )  # fmt: skip

        # ==================== Main loop: the next NUM_BUFFERS - 1 tiles are in flight ====================
        for i in range(num_tiles - (NUM_BUFFERS - 1)):
            tile_idx = split_tile_start + i
            # Prefetch the scales of the next tile; they hide behind this tile.
            k_scales_next, v_scales_next = _load_kv_scales(
                tile_idx + 1, block_tables_base, last_page, context_length, key_scale, value_scale,
                kv_scale_stride_0, kv_scale_stride_1, kv_head_idx, KV_QUANT_MODE,
                KV_BLOCK_SIZE, TILE_SIZE, QK_LAYOUT,
            )  # fmt: skip
            gl.amd.gfx1250.tdm.async_wait(2 * (NUM_BUFFERS - 1))
            m_i, l_i, acc = _process_tile(
                tile_idx, i % NUM_BUFFERS, None, None, k_scales, v_scales, q_dot,
                m_i, l_i, acc, k_smem, v_smem, block_tables_base, last_page, context_length,
                window_lo, qk_scale, row_valid, row_pos, per_token_q_scale,
                key_scale_value, value_scale_value, kv_head_idx, query_seq_len,
                COMPUTE_TYPE, BLOCK_M, HEAD_SIZE_POW2, VALUE_HEAD_SIZE_POW2,
                KV_BLOCK_SIZE, TILE_SIZE, K_X, V_X, VALUE_TRANSPOSED,
                QUERY_QUANT_MODE, KV_QUANT_MODE, IS_FP8, FP8_MAX_VALUE, IS_CAUSAL,
                SLIDING_WINDOW, False, True, KV_GATHER,
                QK_LAYOUT, PV_LAYOUT, DOT_K_LAYOUT, DOT_P_LAYOUT, DOT_V_LAYOUT,
            )  # fmt: skip
            if i + NUM_BUFFERS < num_tiles:
                _issue_tile_load(
                    tile_idx + NUM_BUFFERS, i % NUM_BUFFERS, k_desc, v_desc, k_smem,
                    v_smem, block_tables_base, last_page, num_kv_heads, kv_head_idx,
                    KV_BLOCK_SIZE, TILE_SIZE, K_X, V_X, VALUE_TRANSPOSED, KV_GATHER,
                )  # fmt: skip
            k_scales = k_scales_next
            v_scales = v_scales_next

        # ==================== Epilogue: the last NUM_BUFFERS - 1 tiles ====================
        for j in gl.static_range(NUM_BUFFERS - 1):
            i = num_tiles - (NUM_BUFFERS - 1) + j
            if i >= 0:
                k_scales, v_scales = _load_kv_scales(
                    split_tile_start + i, block_tables_base, last_page, context_length, key_scale, value_scale,
                kv_scale_stride_0, kv_scale_stride_1, kv_head_idx, KV_QUANT_MODE,
                KV_BLOCK_SIZE, TILE_SIZE, QK_LAYOUT,
                )  # fmt: skip
                gl.amd.gfx1250.tdm.async_wait(2 * (NUM_BUFFERS - 2 - j))
                # The last tile can hold slots past the context: mask V.
                m_i, l_i, acc = _process_tile(
                    split_tile_start + i, i % NUM_BUFFERS, None, None, k_scales,
                    v_scales, q_dot, m_i, l_i, acc, k_smem, v_smem, block_tables_base, last_page, context_length,
                window_lo, qk_scale, row_valid, row_pos, per_token_q_scale,
                key_scale_value, value_scale_value, kv_head_idx, query_seq_len,
                COMPUTE_TYPE, BLOCK_M, HEAD_SIZE_POW2, VALUE_HEAD_SIZE_POW2,
                KV_BLOCK_SIZE, TILE_SIZE, K_X, V_X, VALUE_TRANSPOSED,
                QUERY_QUANT_MODE, KV_QUANT_MODE, IS_FP8, FP8_MAX_VALUE, IS_CAUSAL,
                SLIDING_WINDOW,
                    j == NUM_BUFFERS - 2, True, KV_GATHER,
                    QK_LAYOUT, PV_LAYOUT, DOT_K_LAYOUT, DOT_P_LAYOUT, DOT_V_LAYOUT,
                )  # fmt: skip
    elif num_tiles > 0:
        # ==================== Small pages: register gather ====================
        # Issue the loads of tile i+1 before the math of tile i.
        k_next, v_next = _gather_kv_tile(
            split_tile_start * TILE_SIZE, key_cache_ptr, value_cache_ptr,
            block_tables_base, last_page, context_length, num_kv_heads,
            kv_head_idx, stride_key_head, stride_value_head, head_size,
            value_head_size, HEAD_SIZE_POW2, VALUE_HEAD_SIZE_POW2,
            KV_BLOCK_SIZE, TILE_SIZE, K_X, V_X, VALUE_TRANSPOSED,
        )  # fmt: skip
        k_scales_next, v_scales_next = _load_kv_scales(
            split_tile_start, block_tables_base, last_page, context_length, key_scale, value_scale,
                kv_scale_stride_0, kv_scale_stride_1, kv_head_idx, KV_QUANT_MODE,
                KV_BLOCK_SIZE, TILE_SIZE, QK_LAYOUT,
        )  # fmt: skip
        for i in range(num_tiles):
            tile_idx = split_tile_start + i
            k_cur = k_next
            v_cur = v_next
            k_scales = k_scales_next
            v_scales = v_scales_next
            next_tile = gl.minimum(tile_idx + 1, split_tile_end - 1)
            k_next, v_next = _gather_kv_tile(
                next_tile * TILE_SIZE, key_cache_ptr, value_cache_ptr,
                block_tables_base, last_page, context_length, num_kv_heads,
                kv_head_idx, stride_key_head, stride_value_head, head_size,
                value_head_size, HEAD_SIZE_POW2, VALUE_HEAD_SIZE_POW2,
                KV_BLOCK_SIZE, TILE_SIZE, K_X, V_X, VALUE_TRANSPOSED,
            )  # fmt: skip
            k_scales_next, v_scales_next = _load_kv_scales(
                next_tile, block_tables_base, last_page, context_length, key_scale, value_scale,
                kv_scale_stride_0, kv_scale_stride_1, kv_head_idx, KV_QUANT_MODE,
                KV_BLOCK_SIZE, TILE_SIZE, QK_LAYOUT,
            )  # fmt: skip
            m_i, l_i, acc = _process_tile(
                tile_idx, 0, gl.convert_layout(k_cur, DOT_K_LAYOUT),
                gl.convert_layout(v_cur, DOT_V_LAYOUT), k_scales, v_scales, q_dot,
                m_i, l_i, acc, None, None, block_tables_base, last_page, context_length,
                window_lo, qk_scale, row_valid, row_pos, per_token_q_scale,
                key_scale_value, value_scale_value, kv_head_idx, query_seq_len,
                COMPUTE_TYPE, BLOCK_M, HEAD_SIZE_POW2, VALUE_HEAD_SIZE_POW2,
                KV_BLOCK_SIZE, TILE_SIZE, K_X, V_X, VALUE_TRANSPOSED,
                QUERY_QUANT_MODE, KV_QUANT_MODE, IS_FP8, FP8_MAX_VALUE, IS_CAUSAL,
                SLIDING_WINDOW, False, False, False,
                QK_LAYOUT, PV_LAYOUT, DOT_K_LAYOUT, DOT_P_LAYOUT, DOT_V_LAYOUT,
            )  # fmt: skip

    # ==================== Output ====================
    out_cols = gl.arange(0, VALUE_HEAD_SIZE_POW2, layout=gl.SliceLayout(0, PV_LAYOUT))
    out_pos = gl.convert_layout(row_pos, ROW_PV)
    out_head = gl.convert_layout(row_head, ROW_PV)
    out_mask = gl.convert_layout(row_valid, ROW_PV)[:, None] & (
        out_cols[None, :] < value_head_size
    )
    if ONE_SHOT:
        if sinks_ptr is not None:
            sinks = gl.load(
                sinks_ptr + kv_head_idx * query_group_size + row_head,
                mask=row_valid,
                other=float("-inf"),
            ).to(gl.float32)
            m_safe = gl.where(m_i == float("-inf"), 0.0, m_i)
            l_i = l_i + gl.exp2(sinks * LOG2E - m_safe)
        l_safe = gl.where(l_i > 0.0, l_i, 1.0)
        acc = acc * gl.convert_layout(1.0 / l_safe, ROW_PV)[:, None]
        out_offsets = (
            sequence_idx.to(gl.int64) * stride_output_0
            + out_pos[:, None] * stride_output_1
            + kv_head_idx * stride_output_2
            + out_head[:, None] * stride_output_3
            + out_cols[None, :]
        )
        gl.store(output_ptr + out_offsets, acc.to(OUTPUT_DTYPE), mask=out_mask)
    else:
        l_safe = gl.where(l_i > 0.0, l_i, 1.0)
        acc = acc * gl.convert_layout(1.0 / l_safe, ROW_PV)[:, None]
        out_offsets = (
            sequence_idx.to(gl.int64) * stride_output_0
            + kv_head_idx * stride_output_1
            + split_idx * stride_output_2
            + gl.convert_layout(out_row, ROW_PV)[:, None] * stride_output_3
            + out_cols[None, :]
        )
        gl.store(output_ptr + out_offsets, acc.to(OUTPUT_DTYPE), mask=out_mask)
        # The reduce kernel expects the max logit in natural-log units.
        ml_offsets = (
            sequence_idx * stride_max_logits_seq
            + kv_head_idx * stride_max_logits_head
            + split_idx * stride_max_logits_part
            + out_row
        )
        gl.store(max_logits_ptr + ml_offsets, m_i * LN2, mask=row_valid)
        gl.store(exp_sums_ptr + ml_offsets, l_i, mask=row_valid)
