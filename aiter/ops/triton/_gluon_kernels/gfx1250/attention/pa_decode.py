# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon gfx1250 paged attention decode kernel (BF16, Phase 1).

Persistent-scheduling (PS) decode kernel for gfx1250 (RDNA4 / MI450) using
WMMA instructions and TDM async loads.  Called from
``aiter/ops/triton/gluon/pa_decode_gluon.py`` when the device is gfx1250.

Supports:
  - BF16 compute (Phase 1)
  - query_length = 1 (standard decode)
  - KV_BLOCK_SIZE in {16, 64, 128, 1024}
  - GQA with group sizes 1-16 (QUERY_GROUP_SIZE_POW2 = 16)
  - ONE_SHOT (single split -> direct output, no reduce)
  - 5D vectorized key cache [num_blocks, num_kv_heads, head_size//x, kv_block_size, x]
  - 4D non-transposed value cache [num_blocks, num_kv_heads, head_size, kv_block_size]

Tile size is min(KV_BLOCK_SIZE, 128) to fit gfx1250 LDS (320 KB) with
double buffering.  The main loop iterates over tokens in TILE_SIZE steps,
with TDM async_load for the next tile overlapping WMMA compute for the
current tile.
"""

import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_pa_decode_gfx1250_repr = make_kernel_repr(
    "_pa_decode_gluon_gfx1250",
    [
        "ONE_QUERY_GROUP_SIZE_POW2",
        "HEAD_SIZE_POW2",
        "KV_BLOCK_SIZE",
        "CONTEXT_PARTITION_SIZE",
        "TILE_SIZE",
        "ONE_SHOT",
    ],
)


@gluon.jit(repr=_pa_decode_gfx1250_repr)
def _pa_decode_gluon_gfx1250(
    exp_sums_ptr,
    max_logits_ptr,
    output_ptr,
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_tables_ptr,
    context_lengths_ptr,
    softmax_scale: float,
    query_scale,
    key_scale,
    value_scale,
    sinks_ptr,
    stride_max_logits_seq: int,
    stride_max_logits_head: int,
    stride_max_logits_part: int,
    stride_output_bs: int,
    stride_output_len: int,
    stride_output_kv_head: int,
    stride_output_group_size: int,
    stride_query_bs: int,
    stride_query_qlen: int,
    stride_query_kv_head: int,
    stride_query_group_size: int,
    stride_key_block: int,
    stride_key_head: int,
    stride_key_head_split: int,
    stride_key_block_elem: int,
    stride_value_block: int,
    stride_value_head: int,
    stride_value_head_size: int,
    stride_block_table_seq: int,
    stride_query_scale_bs: int,
    stride_query_scale_qlen: int,
    stride_query_scale_kv_head: int,
    kv_scale_stride_0: int,
    kv_scale_stride_1: int,
    query_seq_len: int,
    query_group_size: int,
    head_size: int,
    COMPUTE_TYPE: gl.constexpr,
    QUERY_SEQ_LEN_POW2: gl.constexpr,
    ONE_QUERY_GROUP_SIZE_POW2: gl.constexpr,
    HEAD_SIZE_POW2: gl.constexpr,
    KV_BLOCK_SIZE: gl.constexpr,
    CONTEXT_PARTITION_SIZE: gl.constexpr,
    QUERY_QUANT_MODE: gl.constexpr,
    KV_QUANT_MODE: gl.constexpr,
    VALUE_TRANSPOSED: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
    FP8_MAX_VALUE: gl.constexpr,
    SLIDING_WINDOW: gl.constexpr = 0,
    CDNA_VERSION: gl.constexpr = 1250,
    ONE_SHOT: gl.constexpr = False,
    KEY_HEAD_PADDED: gl.constexpr = False,
    VALUE_HEAD_PADDED: gl.constexpr = False,
    IS_HEAD1: gl.constexpr = False,
):
    # ==================== Constants ====================
    WARP_SIZE: gl.constexpr = 32
    LOG2E: gl.constexpr = 1.4426950408889634
    IS_FP8: gl.constexpr = COMPUTE_TYPE.is_fp8()

    QUERY_GROUP_SIZE_POW2: gl.constexpr = (
        QUERY_SEQ_LEN_POW2 * ONE_QUERY_GROUP_SIZE_POW2
    )

    if KV_QUANT_MODE >= 0:
        KV_ELEM_PER_16B: gl.constexpr = 16
    else:
        KV_ELEM_PER_16B: gl.constexpr = 8

    GFX1250_MAX_TILE: gl.constexpr = 128
    TILE_SIZE: gl.constexpr = (
        KV_BLOCK_SIZE if KV_BLOCK_SIZE <= GFX1250_MAX_TILE else GFX1250_MAX_TILE
    )

    K_HEAD_SPLITS: gl.constexpr = HEAD_SIZE_POW2 // KV_ELEM_PER_16B

    # FP8: WMMA [16,16,128/64] k_width=16; BF16: WMMA [16,16,32] k_width=8
    # QK reduction dim = HEAD_SIZE, PV reduction dim = TILE_SIZE — may differ.
    if IS_FP8:
        WMMA_K_QK: gl.constexpr = 128 if HEAD_SIZE_POW2 > 64 else 64
        WMMA_K_PV: gl.constexpr = 128 if TILE_SIZE > 64 else 64
        K_WIDTH_QK: gl.constexpr = 16
        K_WIDTH_PV: gl.constexpr = 16
    else:
        WMMA_K_QK: gl.constexpr = 32
        WMMA_K_PV: gl.constexpr = 32
        K_WIDTH_QK: gl.constexpr = 8
        K_WIDTH_PV: gl.constexpr = 8

    if COMPUTE_TYPE.is_fp8():
        OUTPUT_DTYPE: gl.constexpr = tl.bfloat16
    else:
        OUTPUT_DTYPE: gl.constexpr = COMPUTE_TYPE

    # ==================== WMMA Layout Definitions ====================
    # QK: warps tile cols (token dim) — matches gfx950 gluon and FlyDSL;
    # softmax row-reduce crosses warps (gluon handles via LDS automatically).
    qk_warp_bases: gl.constexpr = [[0, 1], [0, 2]]
    QK_WMMA_LAYOUT: gl.constexpr = gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        instr_shape=[16, 16, WMMA_K_QK],
        warp_bases=qk_warp_bases,
    )
    pv_warp_bases: gl.constexpr = [[0, 1], [0, 2]]
    PV_WMMA_LAYOUT: gl.constexpr = gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        instr_shape=[16, 16, WMMA_K_PV],
        warp_bases=pv_warp_bases,
    )

    dot_q_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=QK_WMMA_LAYOUT, k_width=K_WIDTH_QK
    )
    dot_k_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=QK_WMMA_LAYOUT, k_width=K_WIDTH_QK
    )
    dot_p_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=PV_WMMA_LAYOUT, k_width=K_WIDTH_PV
    )
    dot_v_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=PV_WMMA_LAYOUT, k_width=K_WIDTH_PV
    )

    # ==================== Blocked Layout for Q load / output store ====================
    D_INNER: gl.constexpr = HEAD_SIZE_POW2 // 8
    Q_BLOCKED_LAYOUT: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[WARP_SIZE // (D_INNER // 2), D_INNER // 2],
        warps_per_cta=[2, 2],
        order=[1, 0],
    )

    # ==================== LDS Layouts ====================
    k_shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[HEAD_SIZE_POW2, 8]],
        [TILE_SIZE, HEAD_SIZE_POW2],
        [1, 0],
    )
    v_shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[HEAD_SIZE_POW2, 16]],
        [TILE_SIZE, HEAD_SIZE_POW2],
        [1, 0],
    )

    # ==================== Reduce-friendly layout for softmax ====================
    # Warps split M (rows), lanes split N (cols) → row reduce is warp-local.
    # After QK WMMA (warps along N), convert_layout to this before softmax.
    REDUCE_COLS_PER_LANE: gl.constexpr = TILE_SIZE // 8
    qk_reduce_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, REDUCE_COLS_PER_LANE],
        threads_per_warp=[4, 8],
        warps_per_cta=[4, 1],
        order=[1, 0],
    )

    qk_row_layout: gl.constexpr = gl.SliceLayout(1, qk_reduce_layout)
    pv_row_layout: gl.constexpr = gl.SliceLayout(1, PV_WMMA_LAYOUT)
    qk_col_layout: gl.constexpr = gl.SliceLayout(0, qk_reduce_layout)

    # ==================== Program IDs ====================
    sequence_idx = gl.program_id(0)
    if IS_HEAD1:
        # head_1: kv_head=0, program_id(1) = mtp_idx.
        # Branch A (QUERY_SEQ_LEN_POW2=1): each WG = 1 position, query_begin = mtp_idx.
        # Branch B (QUERY_SEQ_LEN_POW2>1): each WG = multiple positions,
        #   query_begin = mtp_idx * QUERY_SEQ_LEN_POW2.
        kv_head_idx = 0
        query_begin = gl.program_id(1) * QUERY_SEQ_LEN_POW2
    else:
        # General (num_kv_heads>1): no MTP expansion, each WG handles all positions.
        kv_head_idx = gl.program_id(1)
        query_begin = 0
    split_idx = gl.program_id(2)

    context_length = gl.load(context_lengths_ptr + sequence_idx)

    # ==================== Token Range for this split ====================
    # Each split processes a contiguous range of tokens.
    # With sliding window, only attend to [context_length - SLIDING_WINDOW, context_length).
    num_splits = gl.num_programs(2)
    if SLIDING_WINDOW > 0:
        window_start = gl.maximum(0, context_length - SLIDING_WINDOW)
    else:
        window_start = 0
    window_len = context_length - window_start
    tokens_per_split = gl.cdiv(window_len, num_splits)
    token_start = window_start + tokens_per_split * split_idx
    token_end = gl.minimum(context_length, window_start + tokens_per_split * (split_idx + 1))

    # ==================== Q Loading ====================
    # ==================== Row Decomposition (MTP support) ====================
    # Each row in QUERY_GROUP_SIZE_POW2 encodes (query_pos, group_head):
    #   row = query_pos * ONE_QUERY_GROUP_SIZE_POW2 + group_head
    h_offs_q = gl.arange(
        0, QUERY_GROUP_SIZE_POW2, layout=gl.SliceLayout(1, Q_BLOCKED_LAYOUT)
    )
    d_offs = gl.arange(
        0, HEAD_SIZE_POW2, layout=gl.SliceLayout(0, Q_BLOCKED_LAYOUT)
    )
    q_pos_q = h_offs_q // ONE_QUERY_GROUP_SIZE_POW2
    g_head_q = h_offs_q % ONE_QUERY_GROUP_SIZE_POW2
    q_mask = (query_begin + q_pos_q < query_seq_len) & (g_head_q < query_group_size)

    # ==================== Q Loading → LDS → dot operand ====================
    q_base = sequence_idx * stride_query_bs + kv_head_idx * stride_query_kv_head
    q = gl.amd.cdna4.buffer_load(
        ptr=query_ptr + q_base,
        offsets=(
            (query_begin + q_pos_q[:, None]) * stride_query_qlen
            + g_head_q[:, None] * stride_query_group_size
            + d_offs[None, :]
        ).to(gl.int32),
        mask=q_mask[:, None] & (d_offs[None, :] < head_size),
        other=0.0,
    )

    qk_scale = softmax_scale * LOG2E
    if QUERY_QUANT_MODE == 0:
        qk_scale = qk_scale * gl.load(query_scale).to(gl.float32)

    # Store Q to LDS, then reload as WMMA operand layout.
    # Frees VGPRs so Q doesn't stay resident; the loop re-reads from LDS.
    q_shared_layout: gl.constexpr = gl.SwizzledSharedLayout(
        vec=4 if not IS_FP8 else 8, per_phase=1, max_phase=16, order=[1, 0]
    )
    q_shared = gl.allocate_shared_memory(
        COMPUTE_TYPE, [QUERY_GROUP_SIZE_POW2, HEAD_SIZE_POW2], q_shared_layout,
    )
    q_scaled = q.to(gl.float32) * qk_scale
    q_shared.store(q_scaled.to(COMPUTE_TYPE))

    # Per-token query scale (MTP-aware)
    if QUERY_QUANT_MODE == 1:
        q_scale_row = gl.arange(0, QUERY_GROUP_SIZE_POW2, layout=qk_row_layout)
        qs_pos = q_scale_row // ONE_QUERY_GROUP_SIZE_POW2
        qs_head = q_scale_row % ONE_QUERY_GROUP_SIZE_POW2
        q_scale_offsets = (
            sequence_idx * stride_query_scale_bs
            + (query_begin + qs_pos) * stride_query_scale_qlen
            + kv_head_idx * stride_query_scale_kv_head
            + qs_head
        )
        q_scale_mask = (query_begin + qs_pos < query_seq_len) & (qs_head < query_group_size)
        per_token_q_scale = gl.load(
            query_scale + q_scale_offsets, mask=q_scale_mask, other=1.0
        ).to(gl.float32)

    # ==================== Row masks in WMMA layouts ====================
    qk_row_offs = gl.arange(0, QUERY_GROUP_SIZE_POW2, layout=qk_row_layout)
    qk_row_pos = qk_row_offs // ONE_QUERY_GROUP_SIZE_POW2
    qk_row_head = qk_row_offs % ONE_QUERY_GROUP_SIZE_POW2
    qk_row_mask = (query_begin + qk_row_pos < query_seq_len) & (qk_row_head < query_group_size)

    # ==================== Output Offset Setup (MTP-aware) ====================
    h_offs_pv = gl.arange(0, QUERY_GROUP_SIZE_POW2, layout=pv_row_layout)
    d_offs_pv = gl.arange(0, HEAD_SIZE_POW2, layout=gl.SliceLayout(0, PV_WMMA_LAYOUT))
    pv_pos = h_offs_pv // ONE_QUERY_GROUP_SIZE_POW2
    pv_head = h_offs_pv % ONE_QUERY_GROUP_SIZE_POW2
    h_mask_pv = (query_begin + pv_pos < query_seq_len) & (pv_head < query_group_size)

    if ONE_SHOT:
        output_offsets = (
            sequence_idx * stride_output_bs
            + (query_begin + pv_pos[:, None]) * stride_output_len
            + kv_head_idx * stride_output_kv_head
            + pv_head[:, None] * stride_output_group_size
            + d_offs_pv[None, :]
        )
    else:
        # Map padded row to actual (pos, head) index for temporary output
        actual_row = (query_begin + qk_row_pos) * query_group_size + qk_row_head
        output_offsets = (
            sequence_idx * stride_output_bs
            + split_idx * stride_output_kv_head
            + actual_row[:, None] * stride_output_group_size
            + d_offs_pv[None, :]
        )
    output_mask = h_mask_pv[:, None] & (d_offs_pv[None, :] < head_size)

    if not ONE_SHOT:
        actual_row_ml = (query_begin + qk_row_offs // ONE_QUERY_GROUP_SIZE_POW2) * query_group_size + (
            qk_row_offs % ONE_QUERY_GROUP_SIZE_POW2
        )
        max_logits_offsets = (
            sequence_idx * stride_max_logits_seq
            + kv_head_idx * stride_max_logits_head
            + split_idx * stride_max_logits_part
            + actual_row_ml
        )
        ml_mask = qk_row_mask

    # ==================== Softmax State Init ====================
    m_i = gl.full(
        [QUERY_GROUP_SIZE_POW2], float("-inf"), dtype=gl.float32, layout=qk_row_layout,
    )
    l_i = gl.full(
        [QUERY_GROUP_SIZE_POW2], 0.0, dtype=gl.float32, layout=qk_row_layout,
    )
    acc = gl.zeros(
        [QUERY_GROUP_SIZE_POW2, HEAD_SIZE_POW2], dtype=gl.float32, layout=PV_WMMA_LAYOUT,
    )

    # ==================== Early Exit ====================
    if token_start >= token_end:
        if not ONE_SHOT:
            _store_partials(
                m_i, l_i, acc.to(OUTPUT_DTYPE),
                max_logits_ptr, exp_sums_ptr, output_ptr,
                max_logits_offsets, output_offsets,
                ml_mask, output_mask,
            )
        return

    # ==================== KV Scale Setup ====================
    if KV_QUANT_MODE == 0:
        key_scale_value = gl.load(key_scale).to(gl.float32)
        value_scale_value = gl.load(value_scale).to(gl.float32)

    # ==================== TDM Descriptors ====================
    # Key cache 5D [num_blocks, num_kv_heads, head_size//x, kv_block_size, x]
    # 2D view: rows = all head_splits across all (block, head) pairs
    #          cols = kv_block_size * x (contiguous per head_split)
    k_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=key_cache_ptr,
        shape=(
            stride_key_block // stride_key_head_split,
            KV_BLOCK_SIZE * KV_ELEM_PER_16B,
        ),
        strides=(stride_key_head_split, 1),
        block_shape=(K_HEAD_SPLITS, TILE_SIZE * KV_ELEM_PER_16B),
        layout=k_shared_layout,
    )

    # Value cache descriptor:
    # 4D non-transposed: [num_blocks, num_kv_heads, head_size, kv_block_size]
    #   → 2D view (blocks*heads*head_size, kv_block_size), stride=kv_block_size
    # 5D transposed:     [num_blocks, num_kv_heads, kv_block_size//x, head_size, x]
    #   → 2D view (blocks*heads*tok_splits, head_size*x), stride=head_size*x
    #   Same pattern as K cache (contiguous per tok_split row).
    V_ELEM_PER_16B: gl.constexpr = KV_ELEM_PER_16B if VALUE_TRANSPOSED else 1
    V_HEAD_SPLITS: gl.constexpr = KV_BLOCK_SIZE // V_ELEM_PER_16B if VALUE_TRANSPOSED else HEAD_SIZE_POW2

    if VALUE_TRANSPOSED:
        v_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=value_cache_ptr,
            shape=(
                stride_value_block // stride_value_head_size,
                KV_BLOCK_SIZE * V_ELEM_PER_16B if KV_BLOCK_SIZE <= 16
                else stride_value_head_size,
            ),
            strides=(stride_value_head_size, 1),
            block_shape=(
                (1, KV_BLOCK_SIZE * V_ELEM_PER_16B) if KV_BLOCK_SIZE <= 16
                else (TILE_SIZE // V_ELEM_PER_16B, HEAD_SIZE_POW2 * V_ELEM_PER_16B)
            ),
            layout=v_shared_layout,
        )
    elif KV_BLOCK_SIZE <= 16:
        v_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=value_cache_ptr,
            shape=(
                stride_value_block // (head_size * KV_BLOCK_SIZE),
                head_size * KV_BLOCK_SIZE,
            ),
            strides=(head_size * KV_BLOCK_SIZE, 1),
            block_shape=(1, head_size * KV_BLOCK_SIZE),
            layout=v_shared_layout,
        )
    else:
        v_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=value_cache_ptr,
            shape=(
                stride_value_block // stride_value_head_size,
                KV_BLOCK_SIZE,
            ),
            strides=(stride_value_head_size, 1),
            block_shape=(HEAD_SIZE_POW2, TILE_SIZE),
            layout=v_shared_layout,
        )

    # ==================== LDS Allocation (double buffer) ====================
    NUM_BUFFERS: gl.constexpr = 2
    k_smem = gl.allocate_shared_memory(
        key_cache_ptr.dtype.element_ty,
        [NUM_BUFFERS, TILE_SIZE, HEAD_SIZE_POW2],
        k_shared_layout,
    )
    v_smem = gl.allocate_shared_memory(
        value_cache_ptr.dtype.element_ty,
        [NUM_BUFFERS, TILE_SIZE, HEAD_SIZE_POW2],
        v_shared_layout,
    )

    # ==================== Block Table Setup ====================
    block_tables_base = block_tables_ptr + sequence_idx * stride_block_table_seq

    num_iters = gl.cdiv(token_end - token_start, TILE_SIZE)
    num_iters = gl.maximum(num_iters, 0)

    # ==================== Prologue: TDM load first two tiles ====================

    # -- Load tile 0 --
    tile0_token = token_start
    tile0_page = gl.load(block_tables_base + tile0_token // KV_BLOCK_SIZE).to(gl.int64)
    tile0_page_off = tile0_token % KV_BLOCK_SIZE

    k_row_0 = (tile0_page * stride_key_head // stride_key_head_split
               + kv_head_idx * K_HEAD_SPLITS)
    gl.amd.gfx1250.tdm.async_load(
        k_desc,
        [k_row_0.to(gl.int32), (tile0_page_off * KV_ELEM_PER_16B).to(gl.int32)],
        k_smem.index(0),
    )
    if VALUE_TRANSPOSED:
        v_row_0 = (tile0_page * stride_value_block // stride_value_head_size
                   + kv_head_idx * V_HEAD_SPLITS)
        v_col_0 = (tile0_page_off // V_ELEM_PER_16B * HEAD_SIZE_POW2 * V_ELEM_PER_16B
                   if KV_BLOCK_SIZE > 16 else 0)
        gl.amd.gfx1250.tdm.async_load(
            v_desc, [v_row_0.to(gl.int32), v_col_0.to(gl.int32)], v_smem.index(0),
        )
    elif KV_BLOCK_SIZE <= 16:
        v_row_0 = (tile0_page * stride_value_block // (head_size * KV_BLOCK_SIZE)
                   + kv_head_idx)
        gl.amd.gfx1250.tdm.async_load(
            v_desc, [v_row_0.to(gl.int32), 0], v_smem.index(0),
        )
    else:
        v_row_0 = (tile0_page * stride_value_block // stride_value_head_size
                   + kv_head_idx * HEAD_SIZE_POW2)
        gl.amd.gfx1250.tdm.async_load(
            v_desc,
            [v_row_0.to(gl.int32), tile0_page_off.to(gl.int32)],
            v_smem.index(0),
        )

    # -- Load tile 1 (prefetch) --
    if num_iters >= 2:
        tile1_token = token_start + TILE_SIZE
        tile1_page = gl.load(block_tables_base + tile1_token // KV_BLOCK_SIZE).to(gl.int64)
        tile1_page_off = tile1_token % KV_BLOCK_SIZE

        k_row_1 = (tile1_page * stride_key_head // stride_key_head_split
                   + kv_head_idx * K_HEAD_SPLITS)
        gl.amd.gfx1250.tdm.async_load(
            k_desc,
            [k_row_1.to(gl.int32), (tile1_page_off * KV_ELEM_PER_16B).to(gl.int32)],
            k_smem.index(1),
        )
        if VALUE_TRANSPOSED:
            v_row_1 = (tile1_page * stride_value_block // stride_value_head_size
                       + kv_head_idx * V_HEAD_SPLITS)
            v_col_1 = (tile1_page_off // V_ELEM_PER_16B * HEAD_SIZE_POW2 * V_ELEM_PER_16B
                       if KV_BLOCK_SIZE > 16 else 0)
            gl.amd.gfx1250.tdm.async_load(
                v_desc, [v_row_1.to(gl.int32), v_col_1.to(gl.int32)], v_smem.index(1),
            )
        elif KV_BLOCK_SIZE <= 16:
            v_row_1 = (tile1_page * stride_value_block // (head_size * KV_BLOCK_SIZE)
                       + kv_head_idx)
            gl.amd.gfx1250.tdm.async_load(
                v_desc, [v_row_1.to(gl.int32), 0], v_smem.index(1),
            )
        else:
            v_row_1 = (tile1_page * stride_value_block // stride_value_head_size
                       + kv_head_idx * HEAD_SIZE_POW2)
            gl.amd.gfx1250.tdm.async_load(
                v_desc,
                [v_row_1.to(gl.int32), tile1_page_off.to(gl.int32)],
                v_smem.index(1),
            )

    buf_idx: gl.int32 = 0

    # ==================== Main Loop ====================
    gl.assume(num_iters >= 1)
    for i in tl.range(0, num_iters - 1):
        cur_token = token_start + i * TILE_SIZE
        async_buf = 1 - buf_idx

        # Wait for current K/V tile
        gl.amd.gfx1250.tdm.async_wait(1)

        # -- QK WMMA --
        q_dot = q_shared.load(dot_q_layout)
        k_tile = k_smem.index(buf_idx).permute([1, 0]).load(dot_k_layout)
        scores = gl.amd.gfx1250.wmma(
            q_dot,
            k_tile,
            gl.zeros(
                [QUERY_GROUP_SIZE_POW2, TILE_SIZE],
                dtype=gl.float32,
                layout=QK_WMMA_LAYOUT,
            ),
        )
        scores = gl.convert_layout(scores, layout=qk_reduce_layout)

        # -- Boundary + causal mask --
        col_offs = cur_token + gl.arange(0, TILE_SIZE, layout=qk_col_layout)
        if IS_CAUSAL:
            # Each query position t sees: col + (ql-1-t) < context_length
            causal_ext = query_seq_len - 1 - (query_begin + qk_row_pos)
            boundary = (causal_ext[:, None] + col_offs[None, :]) < context_length
            boundary = boundary & qk_row_mask[:, None]
            score_bias = gl.where(boundary, 0.0, float("-inf"))
        else:
            boundary = col_offs < context_length
            score_bias = gl.where(
                gl.convert_layout(boundary, qk_col_layout), 0.0, float("-inf")
            )
            score_bias = score_bias[None, :]

        col_boundary = col_offs < context_length  # 1D col mask for scale loading

        if KV_QUANT_MODE == 0:
            scores = scores * key_scale_value
        elif KV_QUANT_MODE == 1:
            cur_page_id = gl.load(block_tables_base + cur_token // KV_BLOCK_SIZE).to(gl.int64)
            cur_page_off = cur_token % KV_BLOCK_SIZE
            scale_base = cur_page_id * kv_scale_stride_0 + kv_head_idx * kv_scale_stride_1
            tile_scale_offs = scale_base + cur_page_off + gl.arange(0, TILE_SIZE, layout=qk_col_layout)
            tile_k_scale = gl.load(key_scale + tile_scale_offs, mask=col_boundary, other=0.0).to(gl.float32)
            tile_v_scale = gl.load(value_scale + tile_scale_offs, mask=col_boundary, other=0.0).to(gl.float32)
            scores = scores * tile_k_scale[None, :]

        if QUERY_QUANT_MODE == 1:
            scores = scores * per_token_q_scale[:, None]

        scores = scores + score_bias

        # -- Online softmax --
        m_block = gl.max(scores, axis=1)
        m_new = gl.maximum(m_i, m_block)
        m_safe = gl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = gl.exp2(m_i - m_safe)
        p = gl.exp2(scores - m_safe[:, None])
        l_i = l_i * alpha + gl.sum(p, axis=1)

        # -- FP8 PV scale handling --
        probability_scale = 1.0
        if IS_FP8:
            if KV_QUANT_MODE == 0:
                p = p * float(FP8_MAX_VALUE)
                probability_scale = value_scale_value / float(FP8_MAX_VALUE)
            elif KV_QUANT_MODE == 1:
                tile_v_scale_valid = gl.where(
                    gl.convert_layout(col_boundary, qk_col_layout), tile_v_scale, 0.0
                )
                v_scale_max = gl.max(tile_v_scale_valid, axis=0)
                inv_v_max = 1.0 / (v_scale_max + 1e-8)
                fp8_inv = float(FP8_MAX_VALUE) * inv_v_max
                p = tile_v_scale_valid[None, :] * fp8_inv * p
                probability_scale = v_scale_max * (1.0 / float(FP8_MAX_VALUE))

        # -- PV WMMA --
        if VALUE_TRANSPOSED:
            v_tile = (
                v_smem.index(buf_idx)
                .reshape([TILE_SIZE // V_ELEM_PER_16B, V_ELEM_PER_16B, HEAD_SIZE_POW2])
                .permute([0, 2, 1])
                .reshape([TILE_SIZE, HEAD_SIZE_POW2])
                .load(dot_v_layout)
            )
        else:
            v_tile = v_smem.index(buf_idx).load(dot_v_layout)
        p_dot = gl.convert_layout(p.to(COMPUTE_TYPE), dot_p_layout)
        acc = acc * gl.convert_layout(alpha[:, None], layout=PV_WMMA_LAYOUT)
        pv_result = gl.amd.gfx1250.wmma(
            p_dot, v_tile,
            gl.zeros([QUERY_GROUP_SIZE_POW2, HEAD_SIZE_POW2], dtype=gl.float32, layout=PV_WMMA_LAYOUT),
        )
        if KV_QUANT_MODE >= 0:
            acc = acc + probability_scale * pv_result
        else:
            acc = acc + pv_result

        m_i = m_new

        # -- Prefetch tile i+2 into the buffer we just consumed --
        next_token = token_start + (i + 2) * TILE_SIZE
        if next_token < token_end:
            next_page = gl.load(block_tables_base + next_token // KV_BLOCK_SIZE).to(gl.int64)
            next_off = next_token % KV_BLOCK_SIZE

            k_row_n = (next_page * stride_key_head // stride_key_head_split
                       + kv_head_idx * K_HEAD_SPLITS)
            gl.amd.gfx1250.tdm.async_load(
                k_desc,
                [k_row_n.to(gl.int32), (next_off * KV_ELEM_PER_16B).to(gl.int32)],
                k_smem.index(buf_idx),
            )
            if VALUE_TRANSPOSED:
                v_row_n = (next_page * stride_value_block // stride_value_head_size
                           + kv_head_idx * V_HEAD_SPLITS)
                v_col_n = (next_off // V_ELEM_PER_16B * HEAD_SIZE_POW2 * V_ELEM_PER_16B
                           if KV_BLOCK_SIZE > 16 else 0)
                gl.amd.gfx1250.tdm.async_load(
                    v_desc, [v_row_n.to(gl.int32), v_col_n.to(gl.int32)], v_smem.index(buf_idx),
                )
            elif KV_BLOCK_SIZE <= 16:
                v_row_n = (next_page * stride_value_block // (head_size * KV_BLOCK_SIZE)
                           + kv_head_idx)
                gl.amd.gfx1250.tdm.async_load(
                    v_desc, [v_row_n.to(gl.int32), 0], v_smem.index(buf_idx),
                )
            else:
                v_row_n = (next_page * stride_value_block // stride_value_head_size
                           + kv_head_idx * HEAD_SIZE_POW2)
                gl.amd.gfx1250.tdm.async_load(
                    v_desc,
                    [v_row_n.to(gl.int32), next_off.to(gl.int32)],
                    v_smem.index(buf_idx),
                )

        buf_idx = async_buf

    # ==================== Epilogue: process last tile ====================
    gl.amd.gfx1250.tdm.async_wait(0)

    last_token = token_start + (num_iters - 1) * TILE_SIZE

    q_dot = q_shared.load(dot_q_layout)
    k_tile = k_smem.index(buf_idx).permute([1, 0]).load(dot_k_layout)
    scores = gl.amd.gfx1250.wmma(
        q_dot,
        k_tile,
        gl.zeros(
            [QUERY_GROUP_SIZE_POW2, TILE_SIZE],
            dtype=gl.float32,
            layout=QK_WMMA_LAYOUT,
        ),
    )
    scores = gl.convert_layout(scores, layout=qk_reduce_layout)

    col_offs = last_token + gl.arange(0, TILE_SIZE, layout=qk_col_layout)
    col_boundary = col_offs < context_length
    if IS_CAUSAL:
        causal_ext = query_seq_len - 1 - (query_begin + qk_row_pos)
        boundary = (causal_ext[:, None] + col_offs[None, :]) < context_length
        boundary = boundary & qk_row_mask[:, None]
        score_bias = gl.where(boundary, 0.0, float("-inf"))
    else:
        boundary = col_boundary
        score_bias = gl.where(
            gl.convert_layout(boundary, qk_col_layout), 0.0, float("-inf")
        )
        score_bias = score_bias[None, :]

    if KV_QUANT_MODE == 0:
        scores = scores * key_scale_value
    elif KV_QUANT_MODE == 1:
        last_page_id = gl.load(block_tables_base + last_token // KV_BLOCK_SIZE).to(gl.int64)
        last_page_off = last_token % KV_BLOCK_SIZE
        scale_base = last_page_id * kv_scale_stride_0 + kv_head_idx * kv_scale_stride_1
        tile_scale_offs = scale_base + last_page_off + gl.arange(0, TILE_SIZE, layout=qk_col_layout)
        tile_k_scale = gl.load(key_scale + tile_scale_offs, mask=col_boundary, other=0.0).to(gl.float32)
        tile_v_scale = gl.load(value_scale + tile_scale_offs, mask=col_boundary, other=0.0).to(gl.float32)
        scores = scores * tile_k_scale[None, :]

    if QUERY_QUANT_MODE == 1:
        scores = scores * per_token_q_scale[:, None]

    scores = scores + score_bias

    m_block = gl.max(scores, axis=1)
    m_new = gl.maximum(m_i, m_block)
    m_safe = gl.where(m_new == float("-inf"), 0.0, m_new)
    alpha = gl.exp2(m_i - m_safe)
    p = gl.exp2(scores - m_safe[:, None])
    if IS_CAUSAL:
        p = gl.where(boundary, p, 0.0)
    else:
        valid_col = gl.convert_layout(col_boundary, qk_col_layout)
        p = gl.where(valid_col[None, :], p, 0.0)
    l_i = l_i * alpha + gl.sum(p, axis=1)

    probability_scale = 1.0
    if IS_FP8:
        if KV_QUANT_MODE == 0:
            p = p * float(FP8_MAX_VALUE)
            probability_scale = value_scale_value / float(FP8_MAX_VALUE)
        elif KV_QUANT_MODE == 1:
            tile_v_scale_valid = gl.where(
                gl.convert_layout(col_boundary, qk_col_layout), tile_v_scale, 0.0
            )
            v_scale_max = gl.max(tile_v_scale_valid, axis=0)
            inv_v_max = 1.0 / (v_scale_max + 1e-8)
            fp8_inv = float(FP8_MAX_VALUE) * inv_v_max
            p = tile_v_scale_valid[None, :] * fp8_inv * p
            probability_scale = v_scale_max * (1.0 / float(FP8_MAX_VALUE))

    if VALUE_TRANSPOSED:
        v_tile = (
            v_smem.index(buf_idx)
            .reshape([TILE_SIZE // V_ELEM_PER_16B, V_ELEM_PER_16B, HEAD_SIZE_POW2])
            .permute([0, 2, 1])
            .reshape([TILE_SIZE, HEAD_SIZE_POW2])
            .load(dot_v_layout)
        )
    else:
        v_tile = v_smem.index(buf_idx).load(dot_v_layout)
    p_dot = gl.convert_layout(p.to(COMPUTE_TYPE), dot_p_layout)
    acc = acc * gl.convert_layout(alpha[:, None], layout=PV_WMMA_LAYOUT)
    pv_result = gl.amd.gfx1250.wmma(
        p_dot, v_tile,
        gl.zeros([QUERY_GROUP_SIZE_POW2, HEAD_SIZE_POW2], dtype=gl.float32, layout=PV_WMMA_LAYOUT),
    )
    if KV_QUANT_MODE >= 0:
        acc = acc + probability_scale * pv_result
    else:
        acc = acc + pv_result

    m_i = m_new

    # ==================== Sinks ====================
    if ONE_SHOT and sinks_ptr is not None:
        sink_head_offs = kv_head_idx * query_group_size + qk_row_head
        sink_vals = gl.load(
            sinks_ptr + sink_head_offs, mask=qk_row_mask, other=float("-inf")
        ).to(gl.float32)
        l_i = l_i + tl.math.exp2((sink_vals - m_i) * LOG2E)

    # ==================== Output ====================
    l_safe = gl.where(l_i > 0.0, l_i, 1.0)
    one_over_l = 1.0 / l_safe
    acc = acc * gl.convert_layout(one_over_l[:, None], layout=PV_WMMA_LAYOUT)
    acc_out = acc.to(OUTPUT_DTYPE)

    if ONE_SHOT:
        out_blocked = gl.convert_layout(acc_out, Q_BLOCKED_LAYOUT)
        gl.amd.cdna4.buffer_store(
            out_blocked,
            ptr=output_ptr,
            offsets=output_offsets.to(gl.int32),
            mask=output_mask,
        )
    else:
        _store_partials(
            m_i, l_i, acc_out,
            max_logits_ptr, exp_sums_ptr, output_ptr,
            max_logits_offsets, output_offsets,
            ml_mask, output_mask,
        )


def _store_partials(
    m_i, l_i, acc_bf16,
    max_logits_ptr, exp_sums_ptr, output_ptr,
    max_logits_offsets, output_offsets,
    ml_mask, output_mask,
    _semantic=None,
) -> None:
    gl.amd.cdna4.buffer_store(
        m_i, ptr=max_logits_ptr,
        offsets=max_logits_offsets.to(gl.int32), mask=ml_mask,
    )
    gl.amd.cdna4.buffer_store(
        l_i, ptr=exp_sums_ptr,
        offsets=max_logits_offsets.to(gl.int32), mask=ml_mask,
    )
    gl.amd.cdna4.buffer_store(
        acc_bf16, ptr=output_ptr,
        offsets=output_offsets.to(gl.int32), mask=output_mask,
    )


_store_partials.__triton_builtin__ = True
