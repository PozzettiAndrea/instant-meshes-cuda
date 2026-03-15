/*
    subdivide_gpu.h: GPU-accelerated edge subdivision for triangle meshes

    Ported from QuadriFlow-cuda's GPU subdivision to Instant Meshes.
    Differences from QuadriFlow version:
      - float instead of double (Instant Meshes uses single precision)
      - uint32_t for vertex/face indices (INVALID = 0xFFFFFFFF)
      - No rho sizing field — just maxLength threshold
*/

#pragma once

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

void cuda_subdivide_mesh(
    const float* V_in, uint32_t nV_in,
    const uint32_t* F_in, uint32_t nF_in,
    const uint32_t* E2E_in,
    const uint32_t* boundary_in,
    const uint32_t* nonmanifold_in,
    float maxLength,
    float** V_out, uint32_t* nV_out,
    uint32_t** F_out, uint32_t* nF_out,
    uint32_t** boundary_out,
    uint32_t** nonmanifold_out);

#ifdef __cplusplus
}
#endif
