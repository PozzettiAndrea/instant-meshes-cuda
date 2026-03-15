# Instant Meshes CUDA — Experiments & Findings

## Test Mesh
Stanford Dragon: 437K vertices, 871K faces → 510K vertices after subdivision, target 10K output faces.
GPU: NVIDIA RTX A6000 (84 SMs, 10752 CUDA cores, 48GB VRAM, 768 GB/s DRAM, 6MB L2).

## Final Pipeline (Dragon)

| Stage | Original | Current (hybrid) | Speedup |
|-------|----------|------------------|---------|
| load | 1.8s | 180ms | **10x** |
| subdivide | ~450ms | ~450ms | 1x |
| hierarchy | ~750ms | ~750ms | 1x |
| orient | ~700ms | ~600ms (CPU) | 1.1x |
| position | ~1.5s | ~300ms (GPU, best) | **5x** |
| extract | ~600ms | ~600ms | 1x |
| **Total** | **6.5s** | **2.9s (best), 3.8s (typical)** | **1.7-2.2x** |

## What Worked

### 1. fast_float OBJ Parser (10x load)
Replaced `std::istringstream` + `str_tokenize` + `unordered_map` with bulk `fread` + `fast_float::from_chars` + hand-rolled `parse_int_fast`. The old parser spent 470ns/vertex on `strtof`; fast_float does ~15ns.

**Bug found**: `fast_float::from_chars` does NOT skip leading whitespace (unlike `strtof`). Must manually skip spaces before each call. This caused all vertex coordinates to parse as 0, making scale=0 and subdivision infinite.

### 2. GPU Position Optimization (1.8-5.5x)
The position compat function has ~100 FLOPs/neighbor (4×4 brute force grid search) vs orient's ~30 FLOPs/neighbor. Higher arithmetic intensity makes position compute-bound rather than memory-bound on GPU, crossing the efficiency threshold.

### 3. Spatial BFS Reorder (10-15% all stages)
BFS traversal of faces gives spatially adjacent vertices consecutive indices. Improves L2 cache hit rates for random neighbor access. Original `reorder_mesh` using `unordered_map` took 814ms; hand-written BFS with pre-allocated arrays: 80ms.

### 4. Remove cudaDeviceSynchronize Between Phases (saved ~800ms)
The Gauss-Seidel phases within each iteration must execute sequentially, but CUDA default stream already serializes kernel launches. The explicit sync after each level iteration was pure CPU-GPU round-trip overhead (~0.64ms per sync × 120 syncs = 77ms minimum, plus pipeline drain cost).

### 5. Single .cu File (prevents 4-5x codegen regression)
Having `gpu_hierarchy.cu` and `subdivide_gpu.cu` in the build caused `CUDA_SEPARABLE_COMPILATION` device linking overhead even when set to OFF. The nvcc compiler's cross-TU symbol resolution prevented inlining of device functions, causing position to go from ~300ms to ~1.2s. **All GPU code must be in one .cu file.**

### 6. Pre-cached CSR Adjacency (~100ms saved)
The `cuda_run_optimization` function was converting Link** (pointer-based) to flat CSR arrays on every call. Pre-computing CSR during hierarchy build (when Link** is cache-hot) saves ~150ms of pointer-chasing.

### 7. Branchless Compat Function
Rewrote `compat_orientation_extrinsic_4` to eliminate branches in the inner loop (4 dot products → branchless max). Enables GCC auto-vectorization with `-O3 -march=native`. Marginal improvement (~5%) — compiler was already partially optimizing the branchy version.

## What Failed

### 1. Jacobi Double-Buffer Optimization (DIVERGES)
Replaced Gauss-Seidel phase-sequential updates with Jacobi simultaneous updates (all vertices read from previous iteration, write to new buffer). **Tested**: 12-50 iterations, omega=0.67-0.8. **Result**: 19K-22K singularities vs 160 for Gauss-Seidel. The 4-RoSy compat function makes discrete rotation choices that oscillate under simultaneous updates. The sequential neighbor accumulation in Gauss-Seidel stabilizes these choices.

**Lesson**: Jacobi works for LINEAR systems (Laplacian smoothing) but NOT for non-linear discrete optimization like RoSy field alignment.

### 2. GPU Orient Optimization (3.4x SLOWER than CPU)
The orient compat function has only 1.25 FLOP/byte arithmetic intensity — 40x below the GPU's compute-memory balance point (50 FLOP/byte on A6000). Each vertex reads 6 random neighbors × 24 bytes = 144 bytes of random DRAM access. The GPU's 6MB L2 cache thrashes between Gauss-Seidel phases (each phase writes ~876KB, 7 phases × 876KB = 6.1MB > L2). The CPU's 16-32MB L3 keeps the entire working set warm.

**Nsight profile**: 80% DRAM throughput, 11% compute, 63 registers, 25% achieved occupancy.

### 3. GPU-Resident Hierarchy Build (5x SLOWER)
Attempted to keep all data on GPU: init → dedge → normals → areas → adjacency → downsample × 20 levels → optimize. **Problem**: graph coloring (required for Gauss-Seidel) is inherently sequential → must download adjacency to CPU, color, upload phases back. 60 CPU↔GPU round-trips per hierarchy build. The upload cost (~8ms) was never the bottleneck — the round-trip latency (60 × ~10μs) plus cache invalidation was.

**Lesson**: The upload cost is trivial. Build hierarchy on CPU, upload once.

### 4. Random Partitioning (Vivace-style)
Replaced exact graph coloring with hash-based `group = (id * 2654435761) % 8`. **Result**: hierarchy 38% faster (no coloring cost), but orient/position 80-230% slower (stale reads from non-independent phases reduce convergence quality per iteration). Net: ~same total time.

### 5. Multiple .cu Files (CATASTROPHIC performance regression)
Having `gpu_hierarchy.cu` alongside `optimizer_cuda.cu` caused the position kernel to go from ~300ms to ~1.2s — a 4x regression. **Root cause**: even with `CUDA_SEPARABLE_COMPILATION OFF`, nvcc's device linker resolves `extern __global__` declarations across translation units, preventing inlining of `__device__` helper functions (compat functions, vector math). The position compat function expanded from inline to function-call overhead, increasing register pressure and destroying the instruction scheduler's ability to hide latency.

### 6. `__launch_bounds__` Optimization (NET ZERO)
Forced register count from 63 to 40 via `__launch_bounds__(256, 6)`. Improved occupancy from 25% to 37%. **But**: 23 register spills to local memory added ~1μs per vertex, exactly offsetting the occupancy improvement.

### 7. `__ldg()` Read-Only Cache Hints (4x REGRESSION)
Added `__ldg(&Q[j*3])` for neighbor reads to use the texture cache path. **Result**: 4x slower. The `const float* Q_r = Q` cast introduced pointer aliasing that prevented the compiler from optimizing memory access patterns. NVCC treated `Q_r` and `Q` as potentially different pointers, adding fence instructions.

### 8. L2 Persistent Cache (`cudaStreamSetAttribute`)
Set `hitProp = cudaAccessPropertyPersisting` for Q array. **Result**: no improvement. Q at level 0 is 510K × 12B = 6.1MB — larger than the entire L2 cache (6MB). Persistent cache only helps when the hot dataset is SMALLER than L2.

### 9. Flat Iteration (No Hierarchy)
Ran 1000 Gauss-Seidel iterations at level 0 only, skipping the 20-level hierarchy. **Result**: 2021 singularities (vs 384 with hierarchy). Orient took 7.1s (vs 280ms with hierarchy). The hierarchy's coarse-to-fine propagation is essential for global field consistency — without it, the field gets trapped in local minima where adjacent regions have incompatible orientations.

### 10. Shallow Hierarchy (5 coarsest levels only)
Used top 5 coarse levels + level 0 with extra iterations. **Result**: 518 singularities, 2.4s orient. The INTERMEDIATE levels (3-10) are critical — they bridge the gap between global topology (coarse) and local smoothness (fine). Skipping them degrades quality significantly.

### 11. Persistent Kernel with Cooperative Groups
Fused all 7 phase kernels into one persistent kernel using `grid.sync()` between phases. **Result**: highly variable (254ms to 1.0s). On cold GPU: ~1.0s (no improvement). On warm GPU: ~254ms (great). The variance makes it unreliable. Also adds 7 extra registers for cooperative groups overhead (52 vs 45).

### 12. Adaptive Iteration Count
Reduced iterations at fine levels (4 instead of 6) since propagation provides good initialization. **Result**: position 23% faster (170ms vs 220ms), orient unchanged. Quality identical (±10 singularities).

### 13. Template Kernel Specialization
Compiled separate kernel per `orient_mode` to eliminate switch divergence. **Result**: caused the same linking regression as multiple .cu files — template instantiations in the same TU increased total device code, affecting register allocation for ALL kernels.

## Key Findings

### The Arithmetic Intensity Wall
The orient compat function at 1.25 FLOP/byte is fundamentally unsuited for GPU. The GPU's advantage is raw FLOP throughput, but this kernel is 100% memory-bound. The CPU wins because its cache hierarchy (L3 32MB) is 5x larger than the GPU's (L2 6MB), and TBB's chunk-based processing keeps the working set in L1/L2 per core.

**The GPU only wins on position** because the position compat function has ~4x higher arithmetic intensity (100 FLOPs from 4×4 grid search), shifting the bottleneck partially toward compute where the GPU has an advantage.

### NVCC Codegen Sensitivity
NVCC's register allocator and instruction scheduler are extremely sensitive to the total amount of device code in the compilation unit. Adding unused kernels, unused device functions, or even unused template instantiations can degrade performance of the active kernels by 2-5x through register pressure spillover and reduced inlining.

**Rule**: keep the .cu file minimal. Only include the kernels that are actually called.

### cudaDeviceSynchronize is Poison
Each `cudaDeviceSynchronize` call costs ~0.64ms (measured empirically with dummy kernels). With 120+ syncs in the hierarchical optimization loop, this adds 77ms+ of pure overhead. More importantly, each sync drains the GPU pipeline — no kernel can overlap with the CPU's work of scheduling the next batch.

**Rule**: one sync at the very end. Default stream serializes kernel launches automatically.

## Open Questions / Future Work

1. **RXMesh patch-based processing**: partition mesh into ~256-vertex patches that fit in shared memory. Pre-gather all neighbor data into a contiguous buffer per patch. Converts random reads to coalesced reads — potential 14x bandwidth improvement for orient.

2. **AVX2/AVX-512 vectorized CPU orient**: the branchless compat function is vectorizable. Manual AVX2 intrinsics could achieve 280ms orient (vs 600ms current). Auto-vectorization with branchless code gets ~400ms.

3. **Two-pass GPU position kernel**: split into gather (compute compat, 35 regs) + accumulate (weighted avg, 20 regs). Reduces register pressure from 63 to max(35,20), doubling occupancy.

4. **CPU/GPU overlap**: upload hierarchy async during CPU orient. Position launch overlapped with singularity computation.

5. **Larger meshes**: at 5M+ vertices, the GPU position advantage grows proportionally. The 510K dragon is a poor benchmark for GPU — the working set barely exceeds L2.
