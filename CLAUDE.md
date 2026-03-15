# Instant Meshes CUDA — Developer Guide

## Project Overview

CUDA-accelerated fork of [Instant Meshes](https://github.com/wjakob/instant-meshes) (Jakob et al., SIGGRAPH Asia 2015). Produces quad-dominant meshes from triangle meshes using cross-field optimization + integer grid extraction.

## Build

```bash
# Standard (CPU + CUDA position optimization)
cd build && cmake .. -DBUILD_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
make -j$(nproc)

# With RXMesh GPU subdivision (optional)
scripts/build_rxmesh.sh /usr/local/cuda-13.0 86   # builds ext/rxmesh -> build/rxmesh/
cmake .. -DBUILD_CUDA=ON -DBUILD_RXMESH=ON ...
make -j$(nproc)
```

## Run

```bash
./Instant\ Meshes input.obj -o output.obj -f 10000 -optim cuda
# Options: -optim cpu|cuda, --hier cpu|cuda, --save-dir DIR, --save-all, --run-from/--run-to STAGE
```

## Pipeline Stages (dragon.obj, 437K verts, 10K target faces)

| Stage | Time | GPU? |
|-------|------|------|
| post-load (fast_float OBJ parse) | 180ms | No |
| post-subdivide (edge split + reorder) | 390ms | No (RXMesh experimental) |
| post-hierarchy (multigrid build + coloring) | 650ms | No |
| post-orient (4-RoSy cross-field, GS) | 405ms | No (CPU wins: L3 > L2) |
| post-position (position field, GS) | 523ms | **Yes** |
| post-extract (edge classify + face extract) | 430ms | No |
| **Total** | **~2.8-3.8s** | (was ~6.5s CPU-only) |

## Architecture Decisions

### Single .cu file policy
ALL GPU kernels MUST live in `src/optimizer_cuda.cu`. Having multiple .cu files causes a 2-5x codegen regression — NVCC's register allocator degrades when it sees unused `__global__` functions in the same translation unit. `CUDA_SEPARABLE_COMPILATION` must be `OFF`.

### RXMesh compiled separately
`src/rxmesh_subdivide.cu` is compiled outside CMake via `scripts/build_rxmesh.sh` because:
1. RXMesh needs Eigen 3.4+, nanogui bundles Eigen 3.2 (global include_directories conflict)
2. RXMesh needs `CUDA_SEPARABLE_COMPILATION ON` + device linking, which kills optimizer_cuda.cu perf
3. Pre-compiled .o + device-linked .o avoids contaminating the main build

### CPU orient, GPU position
Orient's compat function has 1.25 FLOP/byte arithmetic intensity (40x below GPU balance). 7 Gauss-Seidel phases write ~876KB each, cycling the entire 6MB L2. CPU's 16-32MB L3 wins. Position has ~4x higher intensity (100 FLOPs from 4x4 grid search), so GPU wins.

### What failed (don't retry these)
- **Jacobi for orient**: Diverges. 19K singularities vs 160 for GS. The discrete rotation choices in compat oscillate under simultaneous updates.
- **Multiple .cu files**: Even unused kernels degrade all other kernels via register pressure spillover.
- **`CUDA_SEPARABLE_COMPILATION ON`**: Prevents cross-TU inlining, 3-10x regression.
- **`__ldg()` intrinsics**: Pointer aliasing breaks compiler optimizations, 4x regression.
- **`cudaDeviceSynchronize` between phases**: 0.64ms each, 139 syncs = 800ms wasted.
- **GPU-resident hierarchy build**: Too many CPU<->GPU round-trips for graph coloring.
- **Penner-inspired position kernel**: Simplified Jacobi+snap produces 43% quads vs 92%.

## Key Files

- `src/optimizer_cuda.cu` — ALL GPU kernels (position optimization, phase-based GS)
- `src/optimizer_cuda.h` — C interface for CUDA functions
- `src/batch.cpp` — Pipeline orchestration, checkpoint save/load, async CPU orient + GPU position
- `src/batch.h` — BatchOptions struct with all CLI flags
- `src/checkpoint.cpp/h` — Binary .imc checkpoint format for pipeline stages
- `src/field.cpp` — Branchless compat_orientation_extrinsic_4()
- `src/hierarchy.cpp/h` — Multigrid hierarchy build with graph coloring
- `src/meshio.cpp` — fast_float OBJ parser (10x over strtof)
- `src/rxmesh_subdivide.cu/h` — RXMesh cavity operator for GPU edge splitting
- `scripts/build_rxmesh.sh` — Builds RXMesh + METIS + device-links
- `ext/rxmesh/` — Vendored RXMesh (CUDA 13 patched)
- `experiments.md` — Detailed log of all tested approaches

## Next Steps (from plan)

1. **Patch-based orient kernel** — pre-gather neighbors into coalesced buffers, 14x bandwidth improvement
2. **Suitor algorithm for hierarchy** — GPU parallel edge merge, replaces sequential greedy (303ms → ~50ms)
3. **JGS2 overshoot correction** — may enable Jacobi-like parallel orient on GPU
4. **GPU edge classification** — extract step 1 is embarrassingly parallel (87ms → ~10ms)
