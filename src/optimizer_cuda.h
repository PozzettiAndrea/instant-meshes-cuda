/*
    optimizer_cuda.h -- C interface to CUDA field optimization kernels

    Provides GPU-accelerated orientation and position field optimization
    for the Instant Meshes multi-resolution hierarchy.
*/

#pragma once

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

struct CUDAOptimizerContext;

CUDAOptimizerContext* cuda_optimizer_create();
void cuda_optimizer_destroy(CUDAOptimizerContext* ctx);

// Upload entire hierarchy to GPU
void cuda_optimizer_upload(CUDAOptimizerContext* ctx,
    int nLevels,
    const uint32_t *level_nVerts,
    const uint32_t *level_nnz,
    const uint32_t **adj_row_ptrs,
    const uint32_t **adj_col_ptrs,
    const float **adj_weight_ptrs,
    const float **V_ptrs,
    const float **N_ptrs,
    const float **Q_ptrs,
    const float **O_ptrs,
    const float **CQ_ptrs,
    const float **CO_ptrs,
    const float **CQw_ptrs,
    const float **COw_ptrs,
    const uint32_t **toUpper_ptrs,
    const uint32_t **phase_index_ptrs,
    const uint32_t *phase_counts,
    const uint32_t *phase_sizes_flat,
    int rosy, int posy, bool extrinsic,
    float scale);

// Run optimization on single level
void cuda_optimize_orientations(CUDAOptimizerContext* ctx, int level);
void cuda_optimize_positions(CUDAOptimizerContext* ctx, int level);

// Propagation
void cuda_propagate_orient(CUDAOptimizerContext* ctx, int fine_level);
void cuda_propagate_pos(CUDAOptimizerContext* ctx, int fine_level);

// Full hierarchical optimization (coarse-to-fine, 6 iterations per level)
void cuda_optimize_orientations_full(CUDAOptimizerContext* ctx, int nLevels);
void cuda_optimize_positions_full(CUDAOptimizerContext* ctx, int nLevels);

// Download fields back to host
void cuda_download_Q(CUDAOptimizerContext* ctx, int level, float *Q_host);
void cuda_download_O(CUDAOptimizerContext* ctx, int level, float *O_host);
void cuda_download_all_fields(CUDAOptimizerContext* ctx,
    float **Q_ptrs, float **O_ptrs, int nLevels);

#ifdef __cplusplus
}
#endif
