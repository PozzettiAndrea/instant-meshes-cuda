/*
    gpu_hierarchy.h -- GPU-resident multi-resolution hierarchy

    Holds all hierarchy data on GPU from mesh init through field optimization.
    Single upload of F,V → build dedge + normals + areas + adjacency + hierarchy
    levels → optimize fields → download final Q,O. One PCIe round-trip for the
    hot path.
*/

#pragma once

#include <cstdint>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct GPUHierarchy;

// Create/destroy
GPUHierarchy* gpu_hierarchy_create();
void gpu_hierarchy_destroy(GPUHierarchy* h);

// Phase 1: Upload F,V → build dedge + normals + areas + adjacency on GPU.
// Downloads V2E, E2E, boundary, nonManifold to CPU (needed for boundary
// alignment and extraction). Geometry stays on GPU.
void gpu_hierarchy_init(GPUHierarchy* h,
    const uint32_t* F, uint32_t nFaces,
    const float* V, uint32_t nVerts,
    uint32_t* h_V2E, uint32_t* h_E2E,
    uint32_t* h_boundary, uint32_t* h_nonManifold);

// Phase 2: Build all hierarchy levels GPU-resident.
// Graph coloring runs on CPU (inherently sequential).
// Returns number of levels built.
int gpu_hierarchy_build(GPUHierarchy* h, bool deterministic);

// Phase 3: Init random Q,O fields on GPU for all levels.
void gpu_hierarchy_init_fields(GPUHierarchy* h, float scale);

// Upload constraints after CPU-side boundary alignment.
void gpu_hierarchy_upload_constraints(GPUHierarchy* h,
    int level, const float* CQ, const float* CO,
    const float* CQw, const float* COw, uint32_t nVerts);

// Run hierarchical field optimization (6 iterations per level, coarse to fine).
void gpu_hierarchy_optimize_orient(GPUHierarchy* h,
    int nLevels, int rosy, int posy, bool extrinsic, float scale);
void gpu_hierarchy_optimize_position(GPUHierarchy* h,
    int nLevels, int rosy, int posy, bool extrinsic, float scale);

// Download fields for extraction / checkpoint.
void gpu_hierarchy_download_Q(GPUHierarchy* h, int level, float* Q, uint32_t nVerts);
void gpu_hierarchy_download_O(GPUHierarchy* h, int level, float* O, uint32_t nVerts);

// Download geometry for checkpoint.
void gpu_hierarchy_download_V(GPUHierarchy* h, int level, float* V, uint32_t nVerts);
void gpu_hierarchy_download_N(GPUHierarchy* h, int level, float* N, uint32_t nVerts);
void gpu_hierarchy_download_A(GPUHierarchy* h, int level, float* A, uint32_t nVerts);

// Get level info.
int gpu_hierarchy_num_levels(GPUHierarchy* h);
uint32_t gpu_hierarchy_level_nVerts(GPUHierarchy* h, int level);

// Download adjacency CSR for CPU graph coloring.
// Caller must pre-allocate row_ptr[nVerts+1] and col_idx[nnz].
uint32_t gpu_hierarchy_download_adj(GPUHierarchy* h, int level,
    uint32_t* row_ptr, uint32_t* col_idx, float* weights);

// Upload phase arrays (graph coloring results) for a level.
void gpu_hierarchy_upload_phases(GPUHierarchy* h, int level,
    const uint32_t* const* phase_ptrs,
    const uint32_t* phase_sizes,
    uint32_t nPhases);

// Download tree connectivity for checkpoint.
void gpu_hierarchy_download_toUpper(GPUHierarchy* h, int level,
    uint32_t* toUpper, uint32_t nCoarseVerts);
void gpu_hierarchy_download_toLower(GPUHierarchy* h, int level,
    uint32_t* toLower, uint32_t nFineVerts);

#ifdef __cplusplus
}
#endif
