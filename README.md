# Assignment: Blackwell GEMM Kernel Optimization

In this assignment, you will progressively build a high-performance FP16 GEMM kernel for NVIDIA Blackwell (SM100) GPUs using TVM/TIRX. Starting from a minimal single-tile kernel, you will incrementally add optimizations — K-loop accumulation, spatial tiling, TMA async loads, software pipelining, persistent kernels, warp specialization, deeper pipelines, multi-CTA clusters, and multi-consumer parallelism — until you arrive at a fully optimized kernel that matches the structure of production-grade implementations.

**Prerequisites**: Familiarity with CUDA programming concepts (threads, warps, shared memory, synchronization). No prior experience with TIRX or Blackwell-specific features is required — this tutorial covers everything you need.

---

## Background: Blackwell GPU Architecture

Before diving into the code, let's understand the key hardware features of NVIDIA Blackwell (SM100) GPUs that make high-performance GEMM possible. If you're familiar with CUDA, you already know about threads, warps, shared memory, and global memory. Blackwell introduces several new hardware units and memory spaces.

### Memory Hierarchy

Blackwell extends the traditional GPU memory hierarchy with new levels:

```mermaid
flowchart TB
    GMEM["Global Memory (GMEM)<br>Large, high latency"]
    SMEM["Shared Memory (SMEM)<br>Per-CTA, low latency"]
    TMEM["Tensor Memory (TMEM)<br>Per-SM, Tensor Core private"]
    RF["Register File (RF)<br>Per-thread, fastest"]

    GMEM -->|"TMA (async, HW-driven)"| SMEM
    SMEM -->|"tcgen05 MMA (async, HW-driven)"| TMEM
    TMEM -->|"LD_TMEM (warpgroup read)"| RF
    SMEM <-->|"TMA store (async)"| GMEM
```

- **Tensor Memory (TMEM)** is new in Blackwell. It is a high-bandwidth scratchpad memory private to the Tensor Cores. The tcgen05 MMA unit writes its output directly to TMEM (not to registers or shared memory). To read the results, you must explicitly load from TMEM to the register file (`LD_TMEM`).

- **TMEM is not directly addressable** by normal instructions — it is accessed through a special address space with `TLane` (row, mapped to threads) and `TCol` (column) axes. You allocate TMEM via `tcgen05.alloc` and deallocate via `tcgen05.dealloc`.

### TMA (Tensor Memory Accelerator)

TMA is a hardware unit that asynchronously copies rectangular tiles between global memory and shared memory. Key advantages over traditional `__shared__` loads:

- **No thread involvement**: A single thread issues the TMA command; the hardware handles the actual data transfer in the background. Other threads don't need to participate.
- **Swizzled layouts**: TMA understands the memory layout and can perform address swizzling on-the-fly, which is essential for efficient Tensor Core access patterns.
- **Byte counting**: TMA works with mbarriers — when the transfer completes, the hardware automatically signals the barrier. The programmer specifies the expected byte count (`expect_tx`), and the hardware arrives on the barrier when that many bytes have been transferred.

TMA also supports **store** operations: moving data from shared memory back to global memory.

### tcgen05 (Tensor Core MMA)

`tcgen05` is Blackwell's matrix multiply-accumulate (MMA) unit. It reads A and B operands from shared memory and writes the result to tensor memory. Key properties:

- **Asynchronous**: You issue `gemm_async` and the computation runs in the background. The instruction returns immediately.
- **Commit + mbarrier**: After issuing MMA, you call `tcgen05.commit` to tell the hardware "when all pending MMAs are done, signal this mbarrier." This enables the next pipeline stage to know when the result is ready.
- **cta_group**: Controls how many CTAs cooperate on a single MMA operation. With `cta_group=2`, two CTAs contribute their B tiles to a wider matrix multiply.

### mbarrier (Memory Barrier)

mbarriers are hardware synchronization primitives stored in shared memory. They support:

- **arrive**: Signal "I'm done" (decrement a counter). TMA and tcgen05 can arrive on mbarriers directly — no thread needed.
- **wait**: Block until all expected arrivals have occurred.
- **Phase flipping**: mbarriers alternate between phase 0 and phase 1, allowing the same barrier to be reused across pipeline stages without confusion.

This is the key to overlapping computation with memory transfers: TMA automatically arrives on a barrier when data is ready, and the MMA warp waits on that barrier before computing.

### Warpgroups

A warpgroup consists of 4 consecutive warps (128 threads). TIRX organizes the thread hierarchy as:

```
kernel > cluster > CTA > warpgroup > warp > thread
```

Warpgroup-level operations include TMEM reads (all 128 threads in a warpgroup participate) and warpgroup-level synchronization.

### CTA Clusters

Blackwell supports **CTA clusters** — groups of CTAs that can cooperate via:

- **shared::cluster memory**: CTAs in the same cluster can access each other's shared memory.
- **Multicast TMA**: A single TMA command can deliver the same data to multiple CTAs simultaneously, reducing global memory bandwidth.
- **Cross-CTA barrier signaling**: mbarrier arrive/wait across CTAs in the cluster.

For GEMM, clustering allows the `cta_group=2` MMA to cross-read B from both CTAs' shared memory via the cluster address space. Each CTA loads its own A and B tiles independently, but the MMA hardware combines both CTAs' B into a wider output, effectively doubling the compute-to-memory ratio.

---

## TIRX Primer

TIRX is an extended Tensor IR built on top of TVM. It provides a Python DSL for writing GPU kernels that map directly to hardware features. Here is the anatomy of a minimal TIRX kernel:

```python
@Tx.prim_func(tirx=True)                    # Declare a TIRX primitive function
def kernel(A: Tx.Buffer((M, K), "float16"),  # Typed buffer parameters
           D: Tx.Buffer((M, N), "float16")):
    with Tx.kernel():                        # Kernel execution scope
        bx, by = Tx.cta_id([grid_m, grid_n], parent="kernel")  # CTA indices
        wg_id = Tx.warpgroup_id([1], parent="cta")             # Warpgroup index
        warp_id = Tx.warp_id([4], parent="warpgroup")          # Warp within WG
        lane_id = Tx.thread_id([32], parent="warp")            # Thread within warp

        pool = Tx.PoolAllocator()            # Shared memory allocator
        Asmem = pool.alloc((128, 64), "float16", layout=A_layout)
        pool.commit()                        # Finalize allocation

        Tx.copy(Asmem[:, :], A[...])         # Synchronous copy GMEM -> SMEM
        Tx.gemm_async(tmem, Asmem, Bsmem,    # Async MMA
                       accum=False, dispatch="tcgen05", cta_group=1)
```

Key conventions:
- **Scope nesting**: `Tx.kernel()` > `Tx.cta()` > `Tx.warpgroup()` > `Tx.warp()` > `Tx.thread()` control which threads execute a block.
- **`Tx.meta_var`**: Creates compile-time aliases for expressions (e.g., `m_st = Tx.meta_var(bx * 128)`).
- **`Tx.ptx.*`**: Direct access to PTX intrinsics (mbarrier, tcgen05, fences, etc.).
- **Layouts**: `tma_shared_layout(dtype, SwizzleMode, shape)` creates swizzled layouts for shared memory buffers that are compatible with TMA.


### Axe Layout

This kernel uses **Axe Layout** ([Hou et al., 2026](https://arxiv.org/abs/2601.19092)), a hardware-aware layout abstraction that maps logical tensor coordinates to named physical axes. Instead of manually computing memory addresses or thread-to-element mappings (as in raw CUDA), you declare a layout on each buffer and the compiler generates the correct address arithmetic, TMA descriptors, and LD_TMEM instructions automatically.

**Syntax.** The layout spec `S[shape : stride@axis]` reads as "map each dimension to a named hardware axis":

```python
S[(128, 512) : (1@TLane, 1@TCol)]
#  ^^^  ^^^     ^^^^^^^^  ^^^^^^^^
#  rows cols    row axis  col axis
# "128 rows on TLane, 512 cols on TCol"
```

If no `@axis` is given (just a plain number), it defaults to the memory axis `m`.

**Quick reference — layouts used in this kernel:**

| When you need... | Use this | Example buffers |
|---|---|---|
| Shared memory for TMA | `tma_shared_layout(dtype, SWIZZLE_128B_ATOM, shape)` | `Asmem`, `Bsmem`, `Dsmem` |
| TMEM buffer | `TileLayout(S[(128, 512) : (1@TLane, 1@TCol)])` | `tmem` |
| Register view for warpgroup TMEM read | `TileLayout(S[(128, N) : (1@axis_tid_in_wg, 1)])` | `Dreg_wg` |

- **SMEM layout**: `tma_shared_layout` creates a swizzled layout for bank-conflict-free access. You don't need to understand swizzle internals — just call this helper function with your dtype, swizzle mode, and buffer shape.
- **TMEM layout**: `TLane` and `TCol` are Blackwell Tensor Memory's native 2D addressing axes. Declaring this layout tells the compiler the buffer lives in TMEM.
- **Register view**: `axis_tid_in_wg` means "distribute rows across the 128 threads in a warpgroup." When you write `Tx.copy(Dreg_wg, tmem)`, the compiler matches `tid_in_wg` to `TLane` and generates the correct LD_TMEM instructions.


---

## Setup

### Modal Setup

1. Install Modal and authenticate with your Andrew email account:

```bash
pip install modal
modal setup
```

2. Run tests via Modal:

```bash
# Run all tests
modal run run_modal.py
# Run a specific step
modal run run_modal.py --step 3
# Run multiple specific steps
modal run run_modal.py --step 1,3,5
```

---

### Local Setup

#### Prerequisites

- **OS**: Linux (Ubuntu 20.04+ recommended)
- **GPU**: NVIDIA Blackwell (B200 / B100) with driver >= 570
- **Python**: >= 3.10 with `pip`

#### Install

```bash
python -m pip install --pre -U -f https://mlc.ai/wheels "mlc-ai-tirx-cu130==0.0.1b2"
pip install torch==2.9.1+cu130 --index-url https://download.pytorch.org/whl/cu130
pip install pytest numpy
```

#### Verify Installation

```bash
python -c "import tvm; print(tvm.__version__)"
python -c "from tvm.script import tirx as Tx; print('TIRX OK')"
```

Both commands should complete without errors.

#### GPU Selection

On multi-GPU machines, select an idle GPU to avoid conflicts with other users:

```bash
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t',' -k2 -n | head -1 | cut -d',' -f1 | tr -d ' ')
```

The test framework (`conftest.py`) also auto-selects the least busy GPU if `CUDA_VISIBLE_DEVICES` is not set, but setting it explicitly is recommended.

If tests fail intermittently, check `nvidia-smi` — another process may be using the GPU. Switch to an idle one.

---

## File Structure

```
gemm_kernels.py          # Skeleton — your implementation goes here
utils.py                 # Helpers: prepare_data, compile_and_run, verify, benchmark
tests/
  conftest.py            # Pytest GPU selection fixture
  test_step01.py         # Step 1 test
  ...
  test_step10.py         # Step 10 test
```

---

## How to Work

1. Open `gemm_kernels.py` and implement the `TODO` sections for each step.
2. Run the corresponding test to verify correctness:
   - **Via Modal (cloud B200):**
     ```bash
     modal run run_modal.py --step XX
     # or run multiple steps at once
     modal run run_modal.py --step 1,3,5
     ```
   - **Locally:**
     ```bash
     python -m pytest tests/test_stepXX.py -xvs
     ```
3. Move on to the next step only after the current one passes.
4. Steps are cumulative — each step builds on the previous one. Read the full step description before starting.

---

## Steps

### Step 1: Single-Tile Synchronous GEMM (Warm-up)

**What you will learn:**
- The basic structure of a TIRX kernel: function declaration, thread hierarchy, memory allocation
- Synchronous data loading (GMEM -> SMEM) and tcgen05 MMA invocation
- TMEM allocation/deallocation and writeback (TMEM -> RF -> GMEM)

**Background:**

This is the simplest possible GEMM: the matrix dimensions exactly match one hardware tile (M=128, N=128, K=64), so no tiling or looping is needed. The entire computation is: load A and B into shared memory, run one MMA, read the result from tensor memory to registers, and write to global memory.

The kernel structure is:

1. **Allocate shared memory**: Use `Tx.PoolAllocator()` to allocate `Asmem` (128x64), `Bsmem` (128x64), an mbarrier, and a TMEM address slot.
2. **Allocate TMEM**: `Tx.ptx.tcgen05.alloc(addr, n_cols=512, cta_group=1)` — only warp 0 does this.
3. **Fence + sync**: `fence.proxy_async("shared::cta")`, `fence.mbarrier_init()`, `cta_sync()` — ensure all allocations are visible.
4. **Load data**: `Tx.copy(Asmem[:,:], A[m_st:m_st+128, 0:64])` — all threads cooperate to copy.
5. **MMA**: Elect one thread via `Tx.ptx.elect_sync()`, call `Tx.gemm_async(tmem[:,:128], Asmem, Bsmem, accum=False, dispatch="tcgen05", cta_group=1)`, then `tcgen05.commit(mma_bar, cta_group=1)`.
6. **Wait for MMA**: `mbarrier.try_wait(mma_bar, phase)`.
7. **Writeback**: Read TMEM to registers (`Tx.copy(Dreg_wg, tmem)` at warpgroup scope), cast fp32 -> fp16, write to GMEM.
8. **Deallocate TMEM**: `tcgen05.relinquish_alloc_permit` + `tcgen05.dealloc`.

**Implementation hints:**
- `accum=False` (not `0`) for the first MMA — TIRX requires a boolean.
- `Tx.copy` at `Tx.cta()` scope means all threads in the CTA cooperate on the copy.
- **Layouts** (see Axe Layout section above):
  - SMEM: `A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))`
  - TMEM: `TileLayout(S[(128, 512) : (1@TLane, 1@TCol)])`
  - Register view for writeback: `TileLayout(S[(128, BLK_N) : (1@axis_tid_in_wg, 1)])`

**Test:** `pytest tests/test_step01.py -xvs`

---

### Step 2: K-Loop Accumulation

**What you will learn:**
- Iterating over the K dimension with multiple MMA invocations
- The `accum` flag: `False` for the first K tile, `True` for subsequent tiles (accumulate into existing TMEM values)
- mbarrier phase flipping for repeated synchronization

**Background:**

Real matrices have K >> 64. To handle this, we loop over K in chunks of `BLK_K=64`. Each iteration loads a new (128x64) A tile and (128x64) B tile, then runs an MMA. The `accum` parameter tells the Tensor Core whether to overwrite TMEM (`False`) or add to it (`True`).

The mbarrier is reused across iterations. After each wait, the phase flips (0 -> 1 -> 0 -> ...) so the next wait doesn't immediately succeed on the old arrival.

**Implementation hints:**
- Loop: `for k in range(K_TILES)` where `K_TILES = K // BLK_K`.
- Load: `Tx.copy(Asmem, A[m_st:m_st+128, k*64:(k+1)*64])`.
- MMA: `accum = (k != 0)` — first iteration is False, rest are True.
- Phase flip: `phase_mma = phase_mma ^ 1` after each wait.

**Test:** `pytest tests/test_step02.py -xvs`

---

### Step 3: Spatial Tiling (Multi-CTA)

**What you will learn:**
- Launching a 2D grid of CTAs to cover arbitrary M and N dimensions
- Per-CTA tile offset calculation

**Background:**

Steps 1-2 only handle M=N=128. To support larger matrices, we launch a 2D grid of CTAs: `[M // BLK_M, N // BLK_N]`. Each CTA computes one 128x128 output tile.

CTA `(bx, by)` computes `D[bx*128 : (bx+1)*128, by*128 : (by+1)*128]` by loading `A[bx*128 : (bx+1)*128, :]` and `B[by*128 : (by+1)*128, :]`.

**Implementation hints:**
- `bx, by = Tx.cta_id([M // BLK_M, N // BLK_N], parent="kernel")`
- `m_st = Tx.meta_var(bx * BLK_M)`, `n_st = Tx.meta_var(by * BLK_N)`
- The K-loop body is the same as step 2, just with offset A and B slices.

**Test:** `pytest tests/test_step03.py -xvs`

---

### Step 4: TMA Async Load

**What you will learn:**
- Replacing synchronous `Tx.copy` with asynchronous TMA: `Tx.copy_async(..., dispatch="tma")`
- Single-thread TMA dispatch via `Tx.ptx.elect_sync()`
- mbarrier-based byte-counting synchronization: `arrive.expect_tx` / `try_wait`
- TMA store writeback: TMEM -> RF -> SMEM -> TMA store -> GMEM

**Background:**

Synchronous loads waste thread resources — all 128 threads sit idle while the memory controller fetches data. TMA offloads this entirely to hardware: one thread issues the command, and the TMA unit handles the rest asynchronously.

The synchronization flow is:

```mermaid
sequenceDiagram
    participant T0 as Thread 0
    participant TMA as TMA Hardware
    participant Bar as mbarrier
    participant All as All Threads
    participant MMA as tcgen05 MMA

    T0->>TMA: copy_async(Asmem, A[...])
    T0->>TMA: copy_async(Bsmem, B[...])
    T0->>Bar: arrive.expect_tx(bytes)
    TMA-->>Bar: auto-arrive when transfer done
    All->>Bar: try_wait(phase)
    Bar-->>All: phase complete
    All->>All: tcgen05.fence.after_thread_sync()
    All->>MMA: gemm_async(tmem, Asmem, Bsmem)
```

The `expect_tx` call tells the mbarrier how many bytes the TMA will transfer. When TMA finishes transferring exactly that many bytes, the barrier automatically transitions.

**Writeback with TMA store:**

Instead of writing directly from registers to GMEM (slow, uncoalesced), we use TMA store:
1. Read TMEM -> registers (LD_TMEM)
2. Cast fp32 -> fp16 in registers
3. Write registers -> Dsmem (shared memory, with swizzled layout)
4. TMA store: `Tx.copy_async(D[...], Dsmem[:,:], dispatch="tma")` — one thread issues TMA store

This requires allocating a `Dsmem` buffer with a TMA-compatible swizzled layout (`tma_shared_layout` — same helper used for Asmem/Bsmem).

**Implementation hints:**
- Only one thread issues TMA: `with Tx.thread(parent="warpgroup")[tid == 0]:`
- TMA config: `{"dispatch": "tma", "cta_group": 1, "mbar": tma_bar.ptr_to([0])}`
- Byte count: `(BLK_M * BLK_K + BLK_N * BLK_K) * 2` (fp16 = 2 bytes)
- Use `Tx.ptx.mbarrier.init(tma_bar.ptr_to([0]), 1)` — 1 expected arrival from the expect_tx call.

**Test:** `pytest tests/test_step04.py -xvs`

---

### Step 5: Software Pipeline (PIPE_DEPTH=2)

**What you will learn:**
- Overlapping TMA loads with MMA computation using double buffering
- Multi-buffered shared memory: `Asmem[stage, :, :]`
- Per-stage phase tracking
- Prefetch loop pattern

**Background:**

Without pipelining, the kernel alternates between loading and computing:

```mermaid
gantt
    title No Pipeline
    dateFormat X
    axisFormat %s
    section TMA
        Load k0 : 0, 2
        Load k1 : 4, 6
        Load k2 : 8, 10
    section MMA
        Compute k0 : 2, 4
        Compute k1 : 6, 8
        Compute k2 : 10, 12
```

With a 2-stage pipeline, we overlap loading the next tile with computing the current one:

```mermaid
gantt
    title PIPE_DEPTH=2
    dateFormat X
    axisFormat %s
    section TMA
        Load k0 : 0, 2
        Load k1 : 2, 4
        Load k2 : 4, 6
    section MMA
        Compute k0 : 2, 4
        Compute k1 : 4, 6
        Compute k2 : 6, 8
```

This requires double-buffered SMEM: `Asmem[0, :, :]` and `Asmem[1, :, :]`. While the MMA reads from stage 0, TMA loads into stage 1, and vice versa.

Each stage has its own mbarrier and phase counter. The pattern is:
1. **Prefetch**: Load the first `PRE_NUM` stages.
2. **Main loop**: For each K tile, wait for load to finish, compute, then issue the next load.

**Implementation hints:**
- `PIPE_DEPTH = 2`
- `Asmem = pool.alloc((PIPE_DEPTH, BLK_M, BLK_K), ...)`
- `tma_bar = pool.alloc((PIPE_DEPTH,), "uint64", ...)`
- Stage tracking: `stage = k % PIPE_DEPTH`
- Per-stage phase: `phase_tma[stage]`

**Test:** `pytest tests/test_step05.py -xvs`

---

### Step 6: Persistent Kernel + Tile Scheduler

**What you will learn:**
- Persistent kernel pattern: fixed number of CTAs that loop over tiles
- `ClusterPersistentScheduler2D` for L2-cache-friendly tile ordering
- Why persistent kernels improve performance

**Background:**

In steps 3-5, each CTA computes exactly one output tile, and the GPU launches `(M/128) * (N/128)` CTAs. For large matrices, this can mean thousands of CTAs, and the launch overhead + cold L2 cache hurt performance.

A persistent kernel launches exactly `SM_COUNT` CTAs (one per SM). Each CTA loops over multiple tiles using a tile scheduler:

```python
tile_scheduler = ClusterPersistentScheduler2D(
    "ts", num_m_tiles=M//128, num_n_tiles=N//128,
    l2_group_size=8, num_clusters=SM_COUNT)
tile_scheduler.init(bx)
while tile_scheduler.valid():
    # ... compute tile at (tile_scheduler.m_idx, tile_scheduler.n_idx) ...
    tile_scheduler.next_tile()
```

The scheduler orders tiles in an L2-cache-friendly pattern (processing nearby tiles together), which significantly improves memory bandwidth utilization.

**Implementation hints:**
- `bx = Tx.cta_id([SM_COUNT], parent="kernel")` — single-dimensional grid.
- `m_st = Tx.meta_var(tile_scheduler.m_idx * BLK_M)`.
- `n_st = Tx.meta_var(tile_scheduler.n_idx * BLK_N)`.
- The K-loop and pipeline logic remain the same as step 5.

**Test:** `pytest tests/test_step06.py -xvs`

---

### Step 7: Warp Specialization (PIPE_DEPTH=2)

**What you will learn:**
- Warp specialization: dedicating different warps/warpgroups to different tasks
- High-level barrier abstractions: `TMABar`, `TCGen05Bar`, `MBarrier`
- `PipelineState` for automatic stage/phase management
- The producer-consumer synchronization chain

**Background:**

This is the biggest architectural change. Instead of all threads doing load-then-compute sequentially, we dedicate specific warps to specific tasks:

- **WG1, warp 3**: TMA producer — continuously loads A and B tiles
- **WG1, warp 0**: MMA consumer — continuously runs MMA as soon as data is ready
- **WG0**: Writeback — reads results from TMEM and writes to GMEM

This requires four types of barriers to synchronize the three actors:

```mermaid
flowchart LR
    TMA["TMA Warp<br>(WG1 warp 3)"]
    MMA["MMA Warp<br>(WG1 warp 0)"]
    WB["Writeback<br>(WG0)"]

    TMA -->|"tma2mma<br>(TMABar: data in SMEM)"| MMA
    MMA -->|"mma2tma<br>(TCGen05Bar: SMEM free)"| TMA
    MMA -->|"mma2ld<br>(TCGen05Bar: result in TMEM)"| WB
    WB -->|"ld2mma<br>(MBarrier: TMEM free)"| MMA
```

- **tma2mma** (`TMABar`): TMA signals MMA "data is in SMEM". TMA hardware auto-arrives via byte counting.
- **mma2tma** (`TCGen05Bar`): MMA signals TMA "SMEM can be reused". tcgen05 hardware auto-arrives via `commit`.
- **mma2ld** (`TCGen05Bar`): MMA signals writeback "results are in TMEM".
- **ld2mma** (`MBarrier`): Writeback signals MMA "TMEM is free for next tile". Threads arrive manually.

`PipelineState` manages stage indices and phase counters automatically:
```python
tma_phase = PipelineState("tma", PIPE_DEPTH)
tma_phase.init(is_producer=True)
# Use tma_phase.stage (current stage index) and tma_phase.phase (current phase)
tma_phase.move_to_next_stage()  # Advance to next stage
```

`TCGen05Bar.arrive` takes a `cta_mask` parameter. For non-cluster kernels (single CTA), use `cta_mask=1`. For cluster kernels (step 9+), use `cta_mask=3` to multicast the signal to both CTAs.

**Epilogue (writeback) structure:**
1. Wait for MMA completion: `mma2ld.wait`
2. Read TMEM to registers in chunks of `TMEM_LD_N=8` columns (hardware bandwidth limit)
3. Cast fp32 -> fp16, accumulate into `Dreg_16b`
4. Signal MMA that TMEM is free: `ld2mma.arrive`
5. Write `Dreg_16b` to `Dsmem` in chunks of `EPI_N=64`, TMA store each chunk to GMEM

**Implementation hints:**
- `WG_NUMBER = 2`, `PIPE_DEPTH = 2`
- Barrier init counts: `tma2mma.init(1)`, `mma2tma.init(1)`, `mma2ld.init(1)`, `ld2mma.init(128)` (all 128 threads in writeback WG arrive)
- TMA warp uses `with Tx.thread(parent="warp")[Tx.ptx.elect_sync()]:` to elect one thread
- MMA warp similarly uses elect_sync
- Both run inside `while tile_scheduler.valid():` loops

**Test:** `pytest tests/test_step07.py -xvs`

---

### Step 8: Deeper Pipeline (PIPE_DEPTH=4)

**What you will learn:**
- The effect of pipeline depth on latency hiding
- How to scale the warp-specialized structure to more pipeline stages

**Background:**

Step 7 uses `PIPE_DEPTH=2` (double buffering). With only 2 stages, the TMA producer can be at most 1 stage ahead of the MMA consumer. If the TMA latency is longer than the MMA compute time, the MMA warp stalls waiting for data.

With `PIPE_DEPTH=4`, the TMA producer can be up to 3 stages ahead, providing more buffering to absorb latency variations. The cost is more shared memory (4x the A/B buffers instead of 2x) and more barrier instances.

**Changes from step 7:**
- `PIPE_DEPTH = 4` (was 2)
- `TMABar(pool, 4, ...)`, `TCGen05Bar(pool, 4, ...)`
- `Asmem = pool.alloc((4, BLK_M, BLK_K), ...)`
- `PipelineState("tma", 4)`, `PipelineState("mma", 4)`

Everything else — the warp specialization structure, barrier flow, and epilogue — remains identical.

**Test:** `pytest tests/test_step08.py -xvs`

---

### Step 9: 2-CTA Cluster

**What you will learn:**
- CTA clusters: multiple CTAs cooperating on a larger tile
- Cross-CTA SMEM access: `cta_group=2` MMA reads B from both CTAs
- Cross-CTA barrier signaling with `cta_mask`
- `cta_group=2` for wider MMA output (MMA_N = BLK_N * CTA_GROUP = 256)

**Background:**

With clustering, two CTAs form a cooperative group. Each CTA has its own shared memory, but they can access each other's SMEM via the `shared::cluster` address space.

The key optimization: with `cta_group=2`, the MMA hardware can read B from **both** CTAs' shared memory via the `shared::cluster` address space. Each CTA loads only 128 columns of B into its own SMEM, but the MMA sees all 256 columns across both CTAs. This doubles the output width without additional memory bandwidth.

```mermaid
flowchart TB
    subgraph GMEM [Global Memory]
        A0["A tile 0<br>(128 x K)"]
        A1["A tile 1<br>(128 x K)"]
        B0["B tile 0<br>(128 x K)"]
        B1["B tile 1<br>(128 x K)"]
    end
    subgraph cluster [CTA Cluster]
        subgraph CTA0 [CTA 0]
            smemA0["Asmem"]
            smemB0["Bsmem<br>B cols 0-127"]
        end
        subgraph CTA1 [CTA 1]
            smemA1["Asmem"]
            smemB1["Bsmem<br>B cols 128-255"]
        end
    end

    A0 -->|TMA load| smemA0
    A1 -->|TMA load| smemA1
    B0 -->|TMA load| smemB0
    B1 -->|TMA load| smemB1
    smemB0 <-.->|"cta_group=2 MMA<br>cross-CTA SMEM read"| smemB1

    subgraph output ["Output per cluster: 256 x 256"]
        D0["CTA 0: D[0:128, 0:256]<br>A0 x combined B"]
        D1["CTA 1: D[128:256, 0:256]<br>A1 x combined B"]
    end
```

The effective output tile per CTA becomes 128 x 256 (instead of 128 x 128), and the cluster tile is 256 x 256. Each CTA loads its own A tile (different M rows) and its own B tile (different N columns). The tcgen05 MMA with `cta_group=2` cross-reads both CTAs' B via `shared::cluster`, producing a 256-column output.

**New concepts:**
- **Cluster CTA ID**: `cbx, cby = Tx.cta_id([CTA_GROUP, 1], parent="cluster")` — position within the cluster.
- **Kernel CTA ID**: `bx = Tx.cta_id([SM_COUNT], parent="kernel")` — which SM.
- **Remote barrier view**: `tma2mma_cta0 = tma2mma.remote_view(0)` — access CTA-0's barrier from any CTA.
- **MMA only on CTA-0**: `if cbx == 0:` — only CTA-0's warp 0 issues MMA commands.
- **Multicast arrive**: `mma2tma.arrive(stage, cta_group=2, cta_mask=3)` — signal both CTAs.
- **TMA arrive only from CTA-0**: `if cbx == 0: tma2mma_cta0.arrive(stage, bytes)`.
- **cluster_sync** instead of **cta_sync** at the end.

**MMA output width:**
With `cta_group=2`, `Tx.gemm_async` outputs `MMA_N = BLK_N * CTA_GROUP = 256` columns (not 128). The epilogue must handle 256 columns of output.

**Implementation hints:**
- `CTA_GROUP = 2`, `MMA_N = BLK_N * CTA_GROUP`
- `m_st` and `n_st` account for cluster position (cbx)
- `ld2mma.init(128 * CTA_GROUP)` — both CTAs' writeback WGs arrive

**Test:** `pytest tests/test_step09.py -xvs`

---

### Step 10: Multi-Consumer Warp Specialization (Final Kernel)

**What you will learn:**
- Multiple MMA warps (consumers) for higher throughput
- Multiple writeback warpgroups
- How the reference production kernel is structured

**Background:**

The final optimization adds a second MMA consumer. With `NUM_CONSUMER=2` and `WG_NUMBER=3`:

- **WG2**: Producer warpgroup
  - warp 0: MMA consumer 0 — computes `A[0:128, :] x B` -> TMEM columns `[0:256]`
  - warp 1: MMA consumer 1 — computes `A[128:256, :] x B` -> TMEM columns `[256:512]`
  - warp 3: TMA producer — loads 2x A blocks + 1x B block per stage
- **WG0**: Writeback for consumer 0 (reads TMEM `[0:256]`)
- **WG1**: Writeback for consumer 1 (reads TMEM `[256:512]`)

This doubles the compute density per CTA: each CTA now processes a 256x256 output tile (vs 128x256 in step 9), and the cluster output becomes 512x256 (vs 256x256 in step 9).

**Changes from step 9:**
- `WG_NUMBER = 3`, `NUM_CONSUMER = 2`
- `Asmem = pool.alloc((PIPE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K), ...)` — 2 A blocks per stage
- TMA loads both `Asmem[stage, 0, :, :]` and `Asmem[stage, 1, :, :]`
- MMA warp `warp_id` selects which A block: `Asmem[stage, warp_id, :, :]`
- MMA output offset: `tmem[:, warp_id * MMA_N : warp_id * MMA_N + MMA_N]`
- Writeback WG offset: `wg_id * MMA_N`
- `mma2tma.init(NUM_CONSUMER)`, `mma2ld.init(1)` per consumer, `ld2mma.init(128 * CTA_GROUP)` per consumer

**Test:** `pytest tests/test_step10.py -xvs`

---

## TIRX API Reference

### Thread Hierarchy

| API | Description |
|-----|-------------|
| `Tx.prim_func(tirx=True)` | Declare a TIRX primitive function |
| `Tx.kernel()` | Kernel execution scope |
| `Tx.cta_id(shape, parent=...)` | CTA index in grid or cluster |
| `Tx.warpgroup_id(shape, parent=...)` | Warpgroup index within CTA |
| `Tx.warp_id(shape, parent=...)` | Warp index within warpgroup |
| `Tx.thread_id(shape, parent=...)` | Thread (lane) index within warp |
| `Tx.ptx.elect_sync()` | Elect one thread in a warp (for single-thread dispatch) |

### Memory

| API | Description |
|-----|-------------|
| `Tx.PoolAllocator()` | Shared memory pool allocator |
| `pool.alloc(shape, dtype, layout=...)` | Allocate buffer from pool |
| `pool.move_base_to(offset)` | Set next allocation offset (for overlapping buffers) |
| `pool.commit()` | Finalize all allocations |
| `Tx.alloc_local(shape, dtype)` | Allocate per-thread register buffer |
| `buf.view(shape, layout=...)` | Create a view of a register buffer with a different layout |  
| `Tx.decl_buffer(shape, dtype, scope="tmem", ...)` | Declare a TMEM buffer |
| `Tx.ptx.tcgen05.alloc(addr, n_cols, cta_group)` | Allocate TMEM |
| `Tx.ptx.tcgen05.dealloc(addr, n_cols, cta_group)` | Deallocate TMEM |

### Data Movement

| API | Description |
|-----|-------------|
| `Tx.copy(dst, src)` | Synchronous copy |
| `Tx.copy_async(dst, src, dispatch="tma", ...)` | TMA async copy (load or store) |
| `Tx.cast(dst, src)` | Element-wise type cast |
| `Tx.gemm_async(C, A, B, accum, dispatch="tcgen05", cta_group)` | tcgen05 MMA |

### Synchronization

| API | Description |
|-----|-------------|
| `Tx.ptx.mbarrier.init(ptr, count)` | Initialize mbarrier with expected arrival count |
| `Tx.ptx.mbarrier.try_wait(ptr, phase)` | Wait for mbarrier phase |
| `Tx.ptx.mbarrier.arrive.expect_tx(ptr, bytes)` | Set expected TMA byte count |
| `Tx.ptx.tcgen05.commit(ptr, cta_group, cta_mask)` | tcgen05 commit (auto-arrive on completion) |
| `Tx.ptx.tcgen05.fence.after_thread_sync()` | Fence before accessing TMEM after sync |
| `Tx.ptx.fence.proxy_async("shared::cta")` | Shared memory fence |
| `Tx.ptx.fence.mbarrier_init()` | Fence after mbarrier initialization |
| `Tx.cuda.cta_sync()` | CTA-wide barrier (like `__syncthreads`) |
| `Tx.cuda.cluster_sync()` | Cluster-wide barrier |

### High-Level Abstractions

| API | Description |
|-----|-------------|
| `TMABar(pool, depth, name)` | TMA barrier array (auto-arrive via byte counting) |
| `TCGen05Bar(pool, depth, name)` | tcgen05 barrier array (auto-arrive via commit) |
| `MBarrier(pool, depth, name)` | Manual mbarrier array (threads arrive explicitly) |
| `PipelineState(name, depth)` | Manages pipeline stage index and phase |
| `ClusterPersistentScheduler2D(...)` | L2-friendly tile scheduler for persistent kernels |

---

## Common Pitfalls

- **Do NOT use Python `and`/`or` on TIR expressions** (e.g., `warp_id == 0 and lane_id == 0`). These are Python operators that don't work on symbolic TIR variables. Use nested `if` statements instead.
- **`accum` must be boolean-compatible**: Use `False` (not `0`) for the first MMA iteration.
- **Fence API**: Use `Tx.ptx.fence.proxy_async("shared::cta")` — positional argument, not keyword `scope=`.
- **GPU flakiness**: If tests fail intermittently, check `nvidia-smi` and switch to an idle GPU.
- **Dsmem overlap**: `pool.move_base_to(1024)` before Dsmem allows it to overlap with Asmem/Bsmem (reusing memory after MMA is done).
- **`alloc_local` vs `decl_buffer`**: Use `Tx.alloc_local` for register buffers. `Tx.decl_buffer` is only for hardware-managed memory like TMEM. To do cross-thread operations, create a view with `.view()` — but use the original `alloc_local` buffer (not the view) for thread-level operations like `Tx.cast`.  

---

## Performance Evaluation

GEMM performance is measured in TFLOPS (Tera Floating-Point Operations Per Second):

```
TFLOPS = 2 * M * N * K / (time_in_seconds) / 1e12
```

The factor of 2 accounts for the multiply and add in each fused multiply-add (FMA) operation.

Use the `benchmark` function in `utils.py` to measure your kernel's performance:

```python
from utils import benchmark
from gemm_kernels import hgemm_v10

kernel = hgemm_v10(4096, 4096, 4096)
ms, tflops = benchmark(kernel, 4096, 4096, 4096)
print(f"{ms:.3f} ms, {tflops:.1f} TFLOPS")
```

---

## Grading Rubric

Total: **100 points**. Each step is graded on correctness, performance (within a bound), and implementation.

| Step | Description | Points |
|------|-------------|--------|
| 1 | Single-tile synchronous GEMM | 5 |
| 2 | K-loop accumulation | 5 |
| 3 | Spatial tiling (multi-CTA) | 10 |
| 4 | TMA async load + TMA store | 10 |
| 5 | Software pipeline (PIPE_DEPTH=2) | 10 |
| 6 | Persistent kernel + tile scheduler | 10 |
| 7 | Warp specialization (PIPE_DEPTH=2) | 15 |
| 8 | Deeper pipeline (PIPE_DEPTH=4) | 5 |
| 9 | 2-CTA cluster | 15 |
| 10 | Multi-consumer (final kernel) | 15 |
| **Total** | | **100** |

---

## Submission

Please follow the instructions carefully.

### What is graded

Each step is graded on two criteria:

1. **Correctness** — the kernel output must match cuBLAS reference (within `rtol=1e-3, atol=1e-2`).
2. **Performance** — the kernel must achieve reasonable TFLOP/s compared to the reference implementation. Kernels significantly slower than the reference will fail the performance check.

### Create submission archive

From your assignment root directory, run:

```bash
tar cvf handin.tar gemm_kernels.py
```

You can verify the contents with:

```bash
tar tvf handin.tar
```

It should list exactly one file:

```
-rw-rw-r-- ... gemm_kernels.py
```

### Submission

**Note: Submissions are not open yet. We will provide the submission details later this week.**
