/*
    init_kernels.cu -- GPU-accelerated mesh initialization for Instant Meshes

    Ported from QuadriFlow-cuda init_kernels.cu, adapted for float precision
    and uint32_t indices. Single GPU-resident pipeline: upload F,V once →
    build dedge (V2E, E2E, boundary, nonManifold) → compute normals →
    compute areas → download all. One PCIe round-trip.
*/

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <cstdio>
#include <cstdint>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#define RCPOVERFLOW_F 2.93873587705571876e-39f
#define INVALID_U 0xFFFFFFFFu

// ============================================================
// Device helpers
// ============================================================

__device__ __host__ inline uint32_t dedge_prev_3(uint32_t e) { return (e % 3 == 0) ? e + 2 : e - 1; }
__device__ __host__ inline uint32_t dedge_next_3(uint32_t e) { return (e % 3 == 2) ? e - 2 : e + 1; }

__device__ inline float d_fast_acos_f(float x) {
    float negate = float(x < 0.0f);
    x = fabsf(x);
    float ret = -0.0187293f;
    ret *= x; ret = ret + 0.0742610f;
    ret *= x; ret = ret - 0.2121144f;
    ret *= x; ret = ret + 1.5707288f;
    ret = ret * sqrtf(1.0f - x);
    ret = ret - 2.0f * negate * ret;
    return negate * M_PI + ret;
}

// Eigen column-major: M(row, col) = M[row + col * 3]
__device__ inline void load_vec3f(const float* M, uint32_t col, float& x, float& y, float& z) {
    uint32_t base = col * 3;
    x = M[base]; y = M[base + 1]; z = M[base + 2];
}

__device__ inline uint32_t load_F(const uint32_t* F, uint32_t row, uint32_t col) {
    return F[row + col * 3];
}

// ============================================================
// Kernel 1: Face normals
// ============================================================

__global__ void k_face_normals(
    const uint32_t* F, const float* V, float* Nf, uint32_t nFaces)
{
    uint32_t f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nFaces) return;

    uint32_t i0 = load_F(F, 0, f), i1 = load_F(F, 1, f), i2 = load_F(F, 2, f);
    float v0x,v0y,v0z, v1x,v1y,v1z, v2x,v2y,v2z;
    load_vec3f(V, i0, v0x,v0y,v0z);
    load_vec3f(V, i1, v1x,v1y,v1z);
    load_vec3f(V, i2, v2x,v2y,v2z);

    float e1x=v1x-v0x, e1y=v1y-v0y, e1z=v1z-v0z;
    float e2x=v2x-v0x, e2y=v2y-v0y, e2z=v2z-v0z;
    float nx=e1y*e2z-e1z*e2y, ny=e1z*e2x-e1x*e2z, nz=e1x*e2y-e1y*e2x;
    float norm = sqrtf(nx*nx + ny*ny + nz*nz);
    if (norm < RCPOVERFLOW_F) { nx=1; ny=0; nz=0; }
    else { float inv=1.0f/norm; nx*=inv; ny*=inv; nz*=inv; }

    uint32_t base = f * 3;
    Nf[base]=nx; Nf[base+1]=ny; Nf[base+2]=nz;
}

// ============================================================
// Kernel 2: Smooth vertex normals (angle-weighted, V2E ring walk)
// ============================================================

__global__ void k_smooth_normals(
    const uint32_t* F, const float* V, const float* Nf,
    const uint32_t* V2E, const uint32_t* E2E,
    const uint32_t* nonManifold,
    float* N, uint32_t nVerts)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nVerts) return;

    uint32_t edge = V2E[i];
    if (nonManifold[i] || edge == INVALID_U) {
        uint32_t base = i * 3;
        N[base]=1; N[base+1]=0; N[base+2]=0;
        return;
    }

    uint32_t stop = edge;
    float normx=0, normy=0, normz=0;
    float vix,viy,viz;
    load_vec3f(V, i, vix,viy,viz);

    do {
        uint32_t idx = edge % 3;
        uint32_t face = edge / 3;
        uint32_t vi1 = load_F(F, (idx+1)%3, face);
        uint32_t vi2 = load_F(F, (idx+2)%3, face);

        float t1x,t1y,t1z, t2x,t2y,t2z;
        load_vec3f(V, vi1, t1x,t1y,t1z);
        load_vec3f(V, vi2, t2x,t2y,t2z);
        float d0x=t1x-vix, d0y=t1y-viy, d0z=t1z-viz;
        float d1x=t2x-vix, d1y=t2y-viy, d1z=t2z-viz;

        float dot = d0x*d1x + d0y*d1y + d0z*d1z;
        float len2_0 = d0x*d0x + d0y*d0y + d0z*d0z;
        float len2_1 = d1x*d1x + d1y*d1y + d1z*d1z;
        float denom = sqrtf(len2_0 * len2_1);
        float angle = (denom > 0) ? d_fast_acos_f(fminf(1.0f, fabsf(dot/denom)) * (dot < 0 ? -1.0f : 1.0f)) : 0;

        if (isfinite(angle)) {
            uint32_t nf_base = face * 3;
            normx += Nf[nf_base]   * angle;
            normy += Nf[nf_base+1] * angle;
            normz += Nf[nf_base+2] * angle;
        }

        uint32_t opp = E2E[edge];
        if (opp == INVALID_U) break;
        edge = dedge_next_3(opp);
    } while (edge != stop);

    float norm = sqrtf(normx*normx + normy*normy + normz*normz);
    uint32_t base = i * 3;
    if (norm > RCPOVERFLOW_F) {
        float inv = 1.0f/norm;
        N[base]=normx*inv; N[base+1]=normy*inv; N[base+2]=normz*inv;
    } else {
        N[base]=1; N[base+1]=0; N[base+2]=0;
    }
}

// ============================================================
// Kernel 3: Vertex area (dual cell area via sub-triangles)
// ============================================================

__global__ void k_vertex_area(
    const uint32_t* F, const float* V,
    const uint32_t* V2E, const uint32_t* E2E,
    const uint32_t* nonManifold,
    float* A, uint32_t nVerts)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nVerts) return;

    uint32_t edge = V2E[i];
    if (nonManifold[i] || edge == INVALID_U) { A[i] = 0; return; }

    uint32_t stop = edge;
    float area = 0;
    float vx,vy,vz;
    load_vec3f(V, load_F(F, edge%3, edge/3), vx,vy,vz);

    do {
        uint32_t ep = dedge_prev_3(edge);
        uint32_t en = dedge_next_3(edge);

        float vnx,vny,vnz, vpx,vpy,vpz;
        load_vec3f(V, load_F(F, en%3, en/3), vnx,vny,vnz);
        load_vec3f(V, load_F(F, ep%3, ep/3), vpx,vpy,vpz);

        float fcx=(vx+vpx+vnx)/3, fcy=(vy+vpy+vny)/3, fcz=(vz+vpz+vnz)/3;
        float px=(vx+vpx)*0.5f, py=(vy+vpy)*0.5f, pz=(vz+vpz)*0.5f;
        float nx=(vx+vnx)*0.5f, ny=(vy+vny)*0.5f, nz=(vz+vnz)*0.5f;

        // area1 = 0.5 * |cross(v-prev_mid, v-face_center)|
        float a1x=vx-px, a1y=vy-py, a1z=vz-pz;
        float b1x=vx-fcx, b1y=vy-fcy, b1z=vz-fcz;
        float cx1=a1y*b1z-a1z*b1y, cy1=a1z*b1x-a1x*b1z, cz1=a1x*b1y-a1y*b1x;
        area += 0.5f * sqrtf(cx1*cx1 + cy1*cy1 + cz1*cz1);

        // area2 = 0.5 * |cross(v-next_mid, v-face_center)|
        float a2x=vx-nx, a2y=vy-ny, a2z=vz-nz;
        float cx2=a2y*b1z-a2z*b1y, cy2=a2z*b1x-a2x*b1z, cz2=a2x*b1y-a2y*b1x;
        area += 0.5f * sqrtf(cx2*cx2 + cy2*cy2 + cz2*cz2);

        uint32_t opp = E2E[edge];
        if (opp == INVALID_U) break;
        edge = dedge_next_3(opp);
    } while (edge != stop);

    A[i] = area;
}

// ============================================================
// Kernel 4-6: Directed edge graph (V2E, E2E, boundary, nonManifold)
// ============================================================

__global__ void k_build_halfedge_lists(
    const uint32_t* F, uint32_t nFaces, uint32_t nVerts,
    uint32_t* V2E, unsigned int* tmp_next, uint32_t* tmp_opposite)
{
    uint32_t f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nFaces) return;

    for (int ii = 0; ii < 3; ++ii) {
        uint32_t idx_cur = load_F(F, ii, f);
        uint32_t idx_next = load_F(F, (ii+1)%3, f);
        uint32_t edge_id = 3 * f + ii;

        if (idx_cur >= nVerts || idx_next >= nVerts || idx_cur == idx_next) continue;

        tmp_opposite[edge_id] = idx_next;
        tmp_next[edge_id] = INVALID_U;

        uint32_t old = atomicCAS(&V2E[idx_cur], INVALID_U, edge_id);
        if (old != INVALID_U) {
            unsigned int idx = old;
            while (true) {
                unsigned int expected = INVALID_U;
                unsigned int prev = atomicCAS(&tmp_next[idx], expected, (unsigned int)edge_id);
                if (prev == expected) break;
                idx = prev;
            }
        }
    }
}

__global__ void k_match_opposites(
    const uint32_t* F, uint32_t nFaces, uint32_t nVerts,
    const uint32_t* V2E, const unsigned int* tmp_next,
    const uint32_t* tmp_opposite, uint32_t* E2E)
{
    uint32_t f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nFaces) return;

    for (int ii = 0; ii < 3; ++ii) {
        uint32_t edge_id_cur = 3 * f + ii;
        uint32_t idx_cur = load_F(F, ii, f);
        uint32_t idx_next = load_F(F, (ii+1)%3, f);
        if (idx_cur == idx_next) continue;

        uint32_t edge_id_opp = V2E[idx_next];
        int found_count = 0;
        uint32_t match = INVALID_U;
        while (edge_id_opp != INVALID_U) {
            if (tmp_opposite[edge_id_opp] == idx_cur) {
                found_count++;
                if (found_count == 1) match = edge_id_opp;
            }
            edge_id_opp = (uint32_t)tmp_next[edge_id_opp];
        }

        if (found_count == 1 && edge_id_cur < match) {
            E2E[edge_id_cur] = match;
            E2E[match] = edge_id_cur;
        }
    }
}

__global__ void k_detect_boundary(
    uint32_t nVerts, uint32_t* V2E, const uint32_t* E2E,
    uint32_t* boundary, uint32_t* nonManifold)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nVerts) return;

    uint32_t edge = V2E[i];
    boundary[i] = 0;
    nonManifold[i] = 0;
    if (edge == INVALID_U) return;

    uint32_t start = edge;
    do {
        uint32_t prev = dedge_prev_3(edge);
        uint32_t opp = E2E[prev];
        if (opp == INVALID_U) {
            boundary[i] = 1;
            V2E[i] = edge;
            return;
        }
        edge = opp;
    } while (edge != start);
}

// ============================================================
// Host wrapper: GPU-resident pipeline
// Upload F,V → build dedge → normals → areas → download
// ============================================================

extern "C" {

void cuda_init_mesh(
    const uint32_t* h_F, uint32_t nFaces,
    const float* h_V, uint32_t nVerts,
    // Outputs (caller allocates)
    uint32_t* h_V2E,          // [nVerts]
    uint32_t* h_E2E,          // [3*nFaces]
    uint32_t* h_boundary,     // [nVerts]
    uint32_t* h_nonManifold,  // [nVerts]
    float* h_N,               // [3*nVerts]
    float* h_A)               // [nVerts]
{
    uint32_t nEdges = 3 * nFaces;
    const int BS = 256;
    int gridF = (nFaces + BS - 1) / BS;
    int gridV = (nVerts + BS - 1) / BS;

    // ---- Allocate all GPU buffers at once ----
    uint32_t *d_F, *d_V2E, *d_E2E, *d_boundary, *d_nonManifold, *d_tmp_opposite;
    unsigned int *d_tmp_next;
    float *d_V, *d_N, *d_Nf, *d_A;

    cudaMalloc(&d_F, nEdges * sizeof(uint32_t));
    cudaMalloc(&d_V, 3 * nVerts * sizeof(float));
    cudaMalloc(&d_V2E, nVerts * sizeof(uint32_t));
    cudaMalloc(&d_E2E, nEdges * sizeof(uint32_t));
    cudaMalloc(&d_boundary, nVerts * sizeof(uint32_t));
    cudaMalloc(&d_nonManifold, nVerts * sizeof(uint32_t));
    cudaMalloc(&d_tmp_next, nEdges * sizeof(unsigned int));
    cudaMalloc(&d_tmp_opposite, nEdges * sizeof(uint32_t));
    cudaMalloc(&d_N, 3 * nVerts * sizeof(float));
    cudaMalloc(&d_Nf, 3 * nFaces * sizeof(float));
    cudaMalloc(&d_A, nVerts * sizeof(float));

    // ---- Single upload ----
    cudaMemcpy(d_F, h_F, nEdges * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, 3 * nVerts * sizeof(float), cudaMemcpyHostToDevice);

    // ---- Phase 1: Build directed edge structure ----
    cudaMemset(d_V2E, 0xFF, nVerts * sizeof(uint32_t));
    cudaMemset(d_E2E, 0xFF, nEdges * sizeof(uint32_t));
    cudaMemset(d_tmp_next, 0xFF, nEdges * sizeof(unsigned int));
    cudaMemset(d_tmp_opposite, 0xFF, nEdges * sizeof(uint32_t));

    k_build_halfedge_lists<<<gridF, BS>>>(d_F, nFaces, nVerts, d_V2E, d_tmp_next, d_tmp_opposite);
    cudaDeviceSynchronize();

    k_match_opposites<<<gridF, BS>>>(d_F, nFaces, nVerts, d_V2E, d_tmp_next, d_tmp_opposite, d_E2E);
    cudaDeviceSynchronize();

    k_detect_boundary<<<gridV, BS>>>(nVerts, d_V2E, d_E2E, d_boundary, d_nonManifold);
    cudaDeviceSynchronize();

    // ---- Phase 2: Face normals + vertex normals ----
    k_face_normals<<<gridF, BS>>>(d_F, d_V, d_Nf, nFaces);
    k_smooth_normals<<<gridV, BS>>>(d_F, d_V, d_Nf, d_V2E, d_E2E, d_nonManifold, d_N, nVerts);
    cudaDeviceSynchronize();

    // ---- Phase 3: Vertex areas ----
    k_vertex_area<<<gridV, BS>>>(d_F, d_V, d_V2E, d_E2E, d_nonManifold, d_A, nVerts);
    cudaDeviceSynchronize();

    // ---- Single download ----
    cudaMemcpy(h_V2E, d_V2E, nVerts * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_E2E, d_E2E, nEdges * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_boundary, d_boundary, nVerts * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_nonManifold, d_nonManifold, nVerts * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_N, d_N, 3 * nVerts * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_A, d_A, nVerts * sizeof(float), cudaMemcpyDeviceToHost);

    // ---- Free all ----
    cudaFree(d_F); cudaFree(d_V);
    cudaFree(d_V2E); cudaFree(d_E2E);
    cudaFree(d_boundary); cudaFree(d_nonManifold);
    cudaFree(d_tmp_next); cudaFree(d_tmp_opposite);
    cudaFree(d_N); cudaFree(d_Nf); cudaFree(d_A);
}

} // extern "C"
