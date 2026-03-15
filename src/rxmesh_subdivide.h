#pragma once
#include <cstdint>
#ifdef __cplusplus
extern "C" {
#endif
void rxmesh_subdivide(
    const float* V_in, uint32_t nV_in,
    const uint32_t* F_in, uint32_t nF_in,
    float maxLength,
    float** V_out, uint32_t* nV_out,
    uint32_t** F_out, uint32_t* nF_out);
#ifdef __cplusplus
}
#endif
