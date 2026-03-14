/*
    init_kernels.h -- C interface to GPU mesh initialization kernels

    GPU-accelerated directed edge structure, smooth normals, and vertex areas.
    Single GPU round-trip: upload F,V → compute all → download results.
*/

#pragma once

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Build directed edge structure + smooth normals + vertex areas on GPU.
// All outputs must be pre-allocated by caller.
void cuda_init_mesh(
    const uint32_t* h_F, uint32_t nFaces,    // Face indices [3*nFaces], column-major
    const float* h_V, uint32_t nVerts,        // Vertex positions [3*nVerts], column-major
    uint32_t* h_V2E,                          // [nVerts] vertex-to-edge
    uint32_t* h_E2E,                          // [3*nFaces] edge-to-edge
    uint32_t* h_boundary,                     // [nVerts] boundary flags
    uint32_t* h_nonManifold,                  // [nVerts] non-manifold flags
    float* h_N,                               // [3*nVerts] vertex normals
    float* h_A);                              // [nVerts] vertex areas

#ifdef __cplusplus
}
#endif
