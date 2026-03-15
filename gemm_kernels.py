import tvm
from tvm.script import tirx as Tx

from tvm.tirx.op_schedule.cuda.common import tma_shared_layout, SwizzleMode
from tvm.tir.layout import TileLayout, S, TLane, TCol, tid_in_wg as axis_tid_in_wg
from tvm.tirx.tile_scheduler import ClusterPersistentScheduler2D
from tvm.tirx.pipeline import PipelineState, MBarrier, TMABar, TCGen05Bar

SM_COUNT = 148  # B200
F16_SIZE = 2

# ======================================================================
# Step 1: Single-tile synchronous GEMM
#   M=128, N=128, K=64 — exactly one tile, no loops.
#   All threads sync-load GMEM→SMEM, one MMA, sync writeback.
# ======================================================================

def hgemm_v1(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_N, BLK_K))

    @Tx.prim_func(tirx=True)
    def kernel(
        A: Tx.Buffer((M, K), a_type),
        B: Tx.Buffer((N, K), b_type),
        D: Tx.Buffer((M, N), d_type),
    ):
        # fmt: off
        with Tx.kernel():
            bx, by = Tx.cta_id([M // BLK_M, N // BLK_N], parent="kernel")
            wg_id = Tx.warpgroup_id([1], parent="cta")
            warp_id = Tx.warp_id([4], parent="warpgroup")
            lane_id = Tx.thread_id([32], parent="warp")

            # --- Shared memory allocation ---
            pool = Tx.PoolAllocator()
            tmem_addr = pool.alloc((1,), "uint32")
            mma_bar = pool.alloc((1,), "uint64", align=8)
            pool.move_base_to(1024)
            Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
            Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
            pool.commit()

            # --- Barrier + TMEM init (warp 0 only) ---
            if warp_id == 0:
                if lane_id == 0:
                    Tx.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
                Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr), n_cols=512, cta_group=1)

            Tx.ptx.fence.proxy_async("shared::cta")
            Tx.ptx.fence.mbarrier_init()
            Tx.cuda.cta_sync()

            tmem = Tx.decl_buffer((128, 512), "float32", scope="tmem", allocated_addr=0,
                                  layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

            m_st = Tx.meta_var(bx * BLK_M)
            n_st = Tx.meta_var(by * BLK_N)

            phase_mma: Tx.int32
            phase_mma = 0

            # TODO: Synchronous load: copy A and B tiles from GMEM to SMEM
            # Hint: use `with Tx.cta():` and `Tx.copy(dst, src)`

            # TODO: Issue MMA (warp 0 only, elected thread)
            # Hint: Tx.gemm_async(tmem[...], Asmem[...], Bsmem[...],
            #          accum=False, dispatch="tcgen05", cta_group=1)
            # Then commit and wait on mma_bar

            # TODO: Writeback: TMEM → RF → GMEM
            # Hint: Tx.copy from tmem to Dreg_wg (with warpgroup view),
            #       Tx.cast to fp16, then Tx.copy to D

            # --- TMEM cleanup ---
            if warp_id == 0:
                Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
                Tx.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel


# ======================================================================
# Step 2: K-loop — accumulate in TMEM
#   M=128, N=128, K=any multiple of 64.
#   Loop over K dimension with accumulation.
# ======================================================================

def hgemm_v2(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_N, BLK_K))

    @Tx.prim_func(tirx=True)
    def kernel(
        A: Tx.Buffer((M, K), a_type),
        B: Tx.Buffer((N, K), b_type),
        D: Tx.Buffer((M, N), d_type),
    ):
        # fmt: off
        with Tx.kernel():
            bx, by = Tx.cta_id([1, 1], parent="kernel")
            wg_id = Tx.warpgroup_id([1], parent="cta")
            warp_id = Tx.warp_id([4], parent="warpgroup")
            lane_id = Tx.thread_id([32], parent="warp")

            pool = Tx.PoolAllocator()
            tmem_addr = pool.alloc((1,), "uint32")
            mma_bar = pool.alloc((1,), "uint64", align=8)
            pool.move_base_to(1024)
            Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
            Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
            pool.commit()

            if warp_id == 0:
                if lane_id == 0:
                    Tx.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
                Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr), n_cols=512, cta_group=1)

            Tx.ptx.fence.proxy_async("shared::cta")
            Tx.ptx.fence.mbarrier_init()
            Tx.cuda.cta_sync()

            tmem = Tx.decl_buffer((128, 512), "float32", scope="tmem", allocated_addr=0,
                                  layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

            phase_mma: Tx.int32
            phase_mma = 0

            # TODO: Loop over K_TILES. For each k:
            #   1. Sync-load A[:, k*BLK_K : (k+1)*BLK_K] and B[:, ...] to SMEM
            #   2. Issue MMA with accum=(k != 0)
            #   3. Wait on mma_bar, flip phase

            # TODO: Writeback TMEM → RF → GMEM (same as step 1)

            if warp_id == 0:
                Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
                Tx.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel


# ======================================================================
# Step 3: Spatial tiling — multi-CTA
#   M, N any multiples of 128, K any multiple of 64.
#   Grid of (M/128)×(N/128) CTAs.
# ======================================================================

def hgemm_v3(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_N, BLK_K))

    @Tx.prim_func(tirx=True)
    def kernel(
        A: Tx.Buffer((M, K), a_type),
        B: Tx.Buffer((N, K), b_type),
        D: Tx.Buffer((M, N), d_type),
    ):
        # fmt: off
        with Tx.kernel():
            # TODO: Launch (M/BLK_M) × (N/BLK_N) CTAs
            # Hint: bx, by = Tx.cta_id([M // BLK_M, N // BLK_N], parent="kernel")
            # Use bx*BLK_M and by*BLK_N as tile offsets.
            # The rest is like step 2 but with dynamic m_st, n_st.
            pass

    return kernel


# ======================================================================
# Step 4: TMA async load
#   Replace sync load with TMA (single-thread dispatch, mbarrier sync).
#   Writeback still sync (TMEM→RF→GMEM).
# ======================================================================

def hgemm_v4(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_N, BLK_K))

    @Tx.prim_func(tirx=True)
    def kernel(
        A: Tx.Buffer((M, K), a_type),
        B: Tx.Buffer((N, K), b_type),
        D: Tx.Buffer((M, N), d_type),
    ):
        # fmt: off
        with Tx.kernel():
            bx, by = Tx.cta_id([M // BLK_M, N // BLK_N], parent="kernel")
            wg_id = Tx.warpgroup_id([1], parent="cta")
            warp_id = Tx.warp_id([4], parent="warpgroup")
            lane_id = Tx.thread_id([32], parent="warp")

            pool = Tx.PoolAllocator()
            tmem_addr = pool.alloc((1,), "uint32")
            tma_bar = pool.alloc((1,), "uint64", align=8)
            mma_bar = pool.alloc((1,), "uint64", align=8)
            pool.move_base_to(1024)
            Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
            Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
            pool.commit()

            if warp_id == 0:
                if lane_id == 0:
                    Tx.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
                    Tx.ptx.mbarrier.init(tma_bar.ptr_to([0]), 1)
                Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr), n_cols=512, cta_group=1)

            Tx.ptx.fence.proxy_async("shared::cta")
            Tx.ptx.fence.mbarrier_init()
            Tx.cuda.cta_sync()

            tmem = Tx.decl_buffer((128, 512), "float32", scope="tmem", allocated_addr=0,
                                  layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

            m_st = Tx.meta_var(bx * BLK_M)
            n_st = Tx.meta_var(by * BLK_N)

            phase_tma: Tx.int32
            phase_mma: Tx.int32
            phase_tma = 0
            phase_mma = 0

            # TODO: Define @Tx.inline tma_load(k_st) that uses:
            #   Tx.copy_async(Asmem, A[...], dispatch="tma", cta_group=1, mbar=...)
            #   Tx.ptx.mbarrier.arrive.expect_tx(tma_bar, byte_count)

            # TODO: Define @Tx.inline mma(accum) that:
            #   1. Waits on tma_bar (data ready)
            #   2. Issues gemm_async + commit
            #   3. Waits on mma_bar (MMA done)

            # TODO: Main loop (elected thread of warp 0):
            #   for k in range(K_TILES): tma_load(k*BLK_K); mma(k != 0)

            # TODO: Writeback TMEM → RF → GMEM

            if warp_id == 0:
                Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
                Tx.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel


# ======================================================================
# Step 5: Software pipeline
#   PIPE_DEPTH=2 multi-buffered SMEM. Prefetch + overlap.
# ======================================================================

def hgemm_v5(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K
    PIPE_DEPTH = 2
    PRE_NUM = min(PIPE_DEPTH, K_TILES)

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_N, BLK_K))

    @Tx.prim_func(tirx=True)
    def kernel(
        A: Tx.Buffer((M, K), a_type),
        B: Tx.Buffer((N, K), b_type),
        D: Tx.Buffer((M, N), d_type),
    ):
        # fmt: off
        with Tx.kernel():
            # TODO: Setup thread hierarchy, allocate PIPE_DEPTH-buffered SMEM,
            # init PIPE_DEPTH mbarriers for TMA, 1 for MMA.
            #
            # Pipeline pattern:
            #   1. Prefetch PRE_NUM stages
            #   2. Main loop: mma(stage) then tma_load(next_stage)
            #   3. Track phase_tma[stage] per stage, phase_mma globally
            #
            # Writeback same as step 4.
            pass

    return kernel


# ======================================================================
# Step 6: Persistent kernel + tile scheduler
#   Fixed SM_COUNT CTAs, loop over tiles with L2-friendly ordering.
# ======================================================================

def hgemm_v6(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K
    PIPE_DEPTH = 2
    PRE_NUM = min(PIPE_DEPTH, K_TILES)

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_N, BLK_K))

    @Tx.prim_func(tirx=True)
    def kernel(
        A: Tx.Buffer((M, K), a_type),
        B: Tx.Buffer((N, K), b_type),
        D: Tx.Buffer((M, N), d_type),
    ):
        # fmt: off
        with Tx.kernel():
            # TODO: Launch SM_COUNT persistent CTAs.
            # Use ClusterPersistentScheduler2D for tile iteration.
            #
            # Key changes from step 5:
            #   - bx = Tx.cta_id([SM_COUNT], parent="kernel")
            #   - tile_scheduler = ClusterPersistentScheduler2D(...)
            #   - while tile_scheduler.valid(): ... tile_scheduler.next_tile()
            #   - m_st/n_st from tile_scheduler.m_idx/n_idx
            pass

    return kernel


# ======================================================================
# Step 7: Warp specialization (PIPE_DEPTH=2)
#   WG1: warp0 (MMA) + warp3 (TMA producer)
#   WG0: writeback (TMEM → RF → SMEM → GMEM)
#   4 barrier types: tma2mma, mma2tma, mma2ld, ld2mma
#   PIPE_DEPTH=2 (same as step 6, focus on warp spec structure)
# ======================================================================

def hgemm_v7(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    MMA_N = BLK_N
    K_TILES = K // BLK_K
    PIPE_DEPTH = 2
    EPI_N = 64
    TMEM_LD_N = 8
    WG_NUMBER = 2

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, EPI_N))

    @Tx.prim_func(tirx=True)
    def kernel(
        A: Tx.Buffer((M, K), a_type),
        B: Tx.Buffer((N, K), b_type),
        D: Tx.Buffer((M, N), d_type),
    ):
        # fmt: off
        with Tx.kernel():
            # TODO: 2 warpgroups, 4 barrier types (TMABar, TCGen05Bar, MBarrier).
            # Use PipelineState for phase tracking.
            #
            # WG1, warp3: TMA producer loop
            #   mma2tma.wait → copy_async A,B → tma2mma.arrive
            #
            # WG1, warp0: MMA consumer loop
            #   ld2mma.wait → K-loop { tma2mma.wait → gemm_async → mma2tma.arrive }
            #   → mma2ld.arrive
            #
            # WG0: writeback loop
            #   mma2ld.wait → TMEM→RF→SMEM→GMEM (TMA store) → ld2mma.arrive
            #
            # Note: use cta_mask=0 for TCGen05Bar.arrive (non-cluster kernel).
            pass

    return kernel


# ======================================================================
# Step 8: Deeper pipeline (PIPE_DEPTH=4)
#   Same warp-specialized structure as v7, but with 4-stage pipeline
#   to better hide TMA latency. Only changes: PIPE_DEPTH=2 → 4,
#   which affects barrier array sizes and Asmem/Bsmem stage dimensions.
# ======================================================================

def hgemm_v8(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    MMA_N = BLK_N
    K_TILES = K // BLK_K
    PIPE_DEPTH = 4
    EPI_N = 64
    TMEM_LD_N = 8
    WG_NUMBER = 2

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, EPI_N))

    @Tx.prim_func(tirx=True)
    def kernel(
        A: Tx.Buffer((M, K), a_type),
        B: Tx.Buffer((N, K), b_type),
        D: Tx.Buffer((M, N), d_type),
    ):
        # fmt: off
        with Tx.kernel():
            # TODO: Same structure as step 7 but with PIPE_DEPTH=4.
            # Changes needed:
            #   - TMABar(pool, 4, ...), TCGen05Bar(pool, 4, ...)
            #   - Asmem/Bsmem shape: (4, BLK_M, BLK_K) / (4, BLK_N, BLK_K)
            #   - PipelineState("tma", 4), PipelineState("mma", 4)
            # Everything else stays the same as step 7.
            pass

    return kernel


# ======================================================================
# Step 9: Cluster — 2-CTA cooperation
#   CTA_GROUP=2, MMA_M=MMA_N=256, cross-CTA TMEM sharing.
# ======================================================================

def hgemm_v9(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    CTA_GROUP = 2
    BLK_M, BLK_N, BLK_K = 128, 128, 64
    MMA_M, MMA_N = 256, 256
    K_TILES = K // BLK_K
    PIPE_DEPTH = 4
    EPI_N = 64
    TMEM_LD_N = 8
    WG_NUMBER = 2
    DTYPE_SIZE = a_type.bits // 8

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, EPI_N))

    @Tx.prim_func(tirx=True)
    def kernel(
        A: Tx.Buffer((M, K), a_type),
        B: Tx.Buffer((N, K), b_type),
        D: Tx.Buffer((M, N), d_type),
    ):
        # fmt: off
        with Tx.kernel():
            # TODO: Extend step 7 with CTA_GROUP=2 cluster.
            # Key changes:
            #   - cbx, cby = Tx.cta_id([CTA_GROUP, 1], parent="cluster")
            #   - tma2mma_cta0 = tma2mma.remote_view(0) for cross-CTA signaling
            #   - MMA output is MMA_N=256 columns (B_N * CTA_GROUP)
            #   - MMA only on cbx==0 (CTA 0 issues MMA for both CTAs)
            #   - cta_mask=3 for TCGen05Bar.arrive (signal both CTAs)
            #   - ld2mma.init(128 * CTA_GROUP) for cross-CTA writeback sync
            #   - Use cluster_sync instead of cta_sync at boundaries
            pass

    return kernel


# ======================================================================
# Step 10: 2-consumer warp specialization
#   NUM_CONSUMER=2, WG2 (TMA+MMA), WG0/WG1 (writeback).
#   This is the final optimized kernel.
# ======================================================================

def hgemm_v10(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    CTA_GROUP = 2
    NUM_CONSUMER = 2
    BLK_M, BLK_N, BLK_K = 128, 128, 64
    MMA_M, MMA_N, MMA_K = 256, 256, 16
    K_TILES = K // BLK_K
    PIPE_DEPTH = 4
    EPI_N = 64
    TMEM_LD_N = 8
    WG_NUMBER = 3
    DTYPE_SIZE = a_type.bits // 8

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM, (NUM_CONSUMER, BLK_M, EPI_N))

    @Tx.prim_func(tirx=True)
    def kernel(
        A: Tx.Buffer((M, K), a_type),
        B: Tx.Buffer((N, K), b_type),
        D: Tx.Buffer((M, N), d_type),
    ):
        # fmt: off
        with Tx.kernel():
            # TODO: 3 warpgroups, 2 consumers, 2-CTA cluster.
            # Key changes from step 9:
            #   - WG_NUMBER=3: WG2 (TMA+MMA), WG0+WG1 (writeback)
            #   - NUM_CONSUMER=2 MMA warps (warp0, warp1 in WG2)
            #   - Each MMA warp handles tmem[:, warp_id*MMA_N : warp_id*MMA_N+MMA_N]
            #   - TMA loads NUM_CONSUMER A blocks per stage
            #   - mma2tma.init(NUM_CONSUMER), mma2ld depth=NUM_CONSUMER
            #   - WG0/WG1 read from tmem offset by wg_id*MMA_N
            #   - Writeback uses per-consumer Dsmem[wg_id, ...]
            pass

    return kernel
