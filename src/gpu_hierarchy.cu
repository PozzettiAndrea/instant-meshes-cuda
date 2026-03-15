/*
    gpu_hierarchy.cu -- GPU-resident multi-resolution hierarchy for Instant Meshes

    All mesh data stays on GPU from init through optimization. Only small
    transfers for CPU graph coloring and greedy merge (both inherently sequential).

    Kernels ported from QuadriFlow-cuda init_kernels.cu (double→float, int→uint32_t).
*/

#include "gpu_hierarchy.h"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#define RCPOVERFLOW_F 2.93873587705571876e-39f
#define INVALID_U 0xFFFFFFFFu
#define BS 256

// ============================================================
// Device helpers
// ============================================================

__device__ __host__ inline uint32_t dedge_prev_3(uint32_t e) { return (e % 3 == 0) ? e + 2 : e - 1; }
__device__ __host__ inline uint32_t dedge_next_3(uint32_t e) { return (e % 3 == 2) ? e - 2 : e + 1; }

__device__ inline float d_fast_acos_f(float x) {
    float negate = float(x < 0.0f);
    x = fabsf(x);
    float ret = -0.0187293f;
    ret *= x; ret += 0.0742610f;
    ret *= x; ret -= 0.2121144f;
    ret *= x; ret += 1.5707288f;
    ret *= sqrtf(1.0f - x);
    ret -= 2.0f * negate * ret;
    return negate * M_PI + ret;
}

__device__ inline void load_vec3f(const float* M, uint32_t col, float& x, float& y, float& z) {
    uint32_t base = col * 3;
    x = M[base]; y = M[base+1]; z = M[base+2];
}

__device__ inline uint32_t load_F(const uint32_t* F, uint32_t row, uint32_t col) {
    return F[row + col * 3];
}

// ============================================================
// Dedge kernels (from init_kernels.cu)
// ============================================================

__global__ void k_build_halfedge_lists(
    const uint32_t* F, uint32_t nFaces, uint32_t nVerts,
    uint32_t* V2E, unsigned int* tmp_next, uint32_t* tmp_opposite)
{
    uint32_t f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nFaces) return;
    for (int ii = 0; ii < 3; ++ii) {
        uint32_t cur = load_F(F, ii, f), nxt = load_F(F, (ii+1)%3, f);
        uint32_t eid = 3*f + ii;
        if (cur >= nVerts || nxt >= nVerts || cur == nxt) continue;
        tmp_opposite[eid] = nxt;
        tmp_next[eid] = INVALID_U;
        uint32_t old = atomicCAS(&V2E[cur], INVALID_U, eid);
        if (old != INVALID_U) {
            unsigned int idx = old;
            while (true) {
                unsigned int exp = INVALID_U;
                unsigned int prev = atomicCAS(&tmp_next[idx], exp, (unsigned int)eid);
                if (prev == exp) break;
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
        uint32_t eid = 3*f + ii;
        uint32_t cur = load_F(F, ii, f), nxt = load_F(F, (ii+1)%3, f);
        if (cur == nxt) continue;
        uint32_t opp = V2E[nxt];
        int found = 0; uint32_t match = INVALID_U;
        while (opp != INVALID_U) {
            if (tmp_opposite[opp] == cur) { found++; if (found == 1) match = opp; }
            opp = (uint32_t)tmp_next[opp];
        }
        if (found == 1 && eid < match) { E2E[eid] = match; E2E[match] = eid; }
    }
}

__global__ void k_detect_boundary(
    uint32_t nVerts, uint32_t* V2E, const uint32_t* E2E,
    uint32_t* boundary, uint32_t* nonManifold)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nVerts) return;
    uint32_t edge = V2E[i];
    boundary[i] = 0; nonManifold[i] = 0;
    if (edge == INVALID_U) return;
    uint32_t start = edge;
    do {
        uint32_t prev = dedge_prev_3(edge);
        uint32_t opp = E2E[prev];
        if (opp == INVALID_U) { boundary[i] = 1; V2E[i] = edge; return; }
        edge = opp;
    } while (edge != start);
}

// ============================================================
// Normal + area kernels
// ============================================================

__global__ void k_face_normals(const uint32_t* F, const float* V, float* Nf, uint32_t nFaces) {
    uint32_t f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nFaces) return;
    uint32_t i0=load_F(F,0,f), i1=load_F(F,1,f), i2=load_F(F,2,f);
    float v0x,v0y,v0z,v1x,v1y,v1z,v2x,v2y,v2z;
    load_vec3f(V,i0,v0x,v0y,v0z); load_vec3f(V,i1,v1x,v1y,v1z); load_vec3f(V,i2,v2x,v2y,v2z);
    float e1x=v1x-v0x,e1y=v1y-v0y,e1z=v1z-v0z, e2x=v2x-v0x,e2y=v2y-v0y,e2z=v2z-v0z;
    float nx=e1y*e2z-e1z*e2y, ny=e1z*e2x-e1x*e2z, nz=e1x*e2y-e1y*e2x;
    float norm=sqrtf(nx*nx+ny*ny+nz*nz);
    if (norm<RCPOVERFLOW_F){nx=1;ny=0;nz=0;} else {float inv=1.f/norm;nx*=inv;ny*=inv;nz*=inv;}
    uint32_t b=f*3; Nf[b]=nx; Nf[b+1]=ny; Nf[b+2]=nz;
}

__global__ void k_smooth_normals(
    const uint32_t* F, const float* V, const float* Nf,
    const uint32_t* V2E, const uint32_t* E2E, const uint32_t* nonManifold,
    float* N, uint32_t nVerts)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nVerts) return;
    uint32_t edge = V2E[i];
    if (nonManifold[i] || edge == INVALID_U) { uint32_t b=i*3; N[b]=1;N[b+1]=0;N[b+2]=0; return; }
    uint32_t stop = edge;
    float normx=0,normy=0,normz=0, vix,viy,viz;
    load_vec3f(V, i, vix,viy,viz);
    do {
        uint32_t idx=edge%3, face=edge/3;
        float t1x,t1y,t1z,t2x,t2y,t2z;
        load_vec3f(V,load_F(F,(idx+1)%3,face),t1x,t1y,t1z);
        load_vec3f(V,load_F(F,(idx+2)%3,face),t2x,t2y,t2z);
        float d0x=t1x-vix,d0y=t1y-viy,d0z=t1z-viz, d1x=t2x-vix,d1y=t2y-viy,d1z=t2z-viz;
        float dot=d0x*d1x+d0y*d1y+d0z*d1z;
        float denom=sqrtf((d0x*d0x+d0y*d0y+d0z*d0z)*(d1x*d1x+d1y*d1y+d1z*d1z));
        float angle=(denom>0)?d_fast_acos_f(fminf(1.f,fabsf(dot/denom))*(dot<0?-1.f:1.f)):0;
        if (isfinite(angle)) { uint32_t nb=face*3; normx+=Nf[nb]*angle; normy+=Nf[nb+1]*angle; normz+=Nf[nb+2]*angle; }
        uint32_t opp=E2E[edge]; if(opp==INVALID_U) break; edge=dedge_next_3(opp);
    } while (edge != stop);
    float norm=sqrtf(normx*normx+normy*normy+normz*normz);
    uint32_t b=i*3;
    if(norm>RCPOVERFLOW_F){float inv=1.f/norm;N[b]=normx*inv;N[b+1]=normy*inv;N[b+2]=normz*inv;}
    else{N[b]=1;N[b+1]=0;N[b+2]=0;}
}

__global__ void k_vertex_area(
    const uint32_t* F, const float* V, const uint32_t* V2E, const uint32_t* E2E,
    const uint32_t* nonManifold, float* A, uint32_t nVerts)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nVerts) return;
    uint32_t edge = V2E[i];
    if (nonManifold[i] || edge == INVALID_U) { A[i] = 0; return; }
    uint32_t stop = edge;
    float area = 0, vx,vy,vz;
    load_vec3f(V, load_F(F, edge%3, edge/3), vx,vy,vz);
    do {
        uint32_t ep=dedge_prev_3(edge), en=dedge_next_3(edge);
        float vnx,vny,vnz,vpx,vpy,vpz;
        load_vec3f(V,load_F(F,en%3,en/3),vnx,vny,vnz);
        load_vec3f(V,load_F(F,ep%3,ep/3),vpx,vpy,vpz);
        float fcx=(vx+vpx+vnx)/3,fcy=(vy+vpy+vny)/3,fcz=(vz+vpz+vnz)/3;
        float a1x=vx-(vx+vpx)*.5f,a1y=vy-(vy+vpy)*.5f,a1z=vz-(vz+vpz)*.5f;
        float b1x=vx-fcx,b1y=vy-fcy,b1z=vz-fcz;
        float cx1=a1y*b1z-a1z*b1y,cy1=a1z*b1x-a1x*b1z,cz1=a1x*b1y-a1y*b1x;
        area+=.5f*sqrtf(cx1*cx1+cy1*cy1+cz1*cz1);
        float a2x=vx-(vx+vnx)*.5f,a2y=vy-(vy+vny)*.5f,a2z=vz-(vz+vnz)*.5f;
        float cx2=a2y*b1z-a2z*b1y,cy2=a2z*b1x-a2x*b1z,cz2=a2x*b1y-a2y*b1x;
        area+=.5f*sqrtf(cx2*cx2+cy2*cy2+cz2*cz2);
        uint32_t opp=E2E[edge]; if(opp==INVALID_U) break; edge=dedge_next_3(opp);
    } while (edge != stop);
    A[i] = area;
}

// ============================================================
// Adjacency CSR from dedge (NEW — replaces CPU generate_adjacency_matrix_uniform)
// ============================================================

__global__ void k_count_adj_from_dedge(
    const uint32_t* F, const uint32_t* V2E, const uint32_t* E2E,
    const uint32_t* nonManifold, uint32_t* counts, uint32_t nVerts)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nVerts) return;
    uint32_t start = V2E[i];
    if (start == INVALID_U || nonManifold[i]) { counts[i] = 0; return; }
    uint32_t count = 0, edge = start;
    do {
        uint32_t opp = E2E[edge];
        uint32_t next = (opp == INVALID_U) ? INVALID_U : dedge_next_3(opp);
        if (count == 0) count++;
        if (opp == INVALID_U || next != start) count++;
        if (opp == INVALID_U) break;
        edge = next;
    } while (edge != start);
    counts[i] = count;
}

__global__ void k_fill_adj_from_dedge(
    const uint32_t* F, const uint32_t* V2E, const uint32_t* E2E,
    const uint32_t* nonManifold, const uint32_t* rowPtr,
    uint32_t* colInd, float* weights, uint32_t nVerts)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nVerts) return;
    uint32_t start = V2E[i];
    if (start == INVALID_U || nonManifold[i]) return;
    uint32_t offset = rowPtr[i], edge = start;
    do {
        uint32_t base = edge%3, f = edge/3;
        uint32_t opp = E2E[edge];
        uint32_t next = (opp == INVALID_U) ? INVALID_U : dedge_next_3(opp);
        if (offset == rowPtr[i]) {
            colInd[offset] = load_F(F, (base+2)%3, f);
            weights[offset] = 1.0f;
            offset++;
        }
        if (opp == INVALID_U || next != start) {
            colInd[offset] = load_F(F, (base+1)%3, f);
            weights[offset] = 1.0f;
            offset++;
            if (opp == INVALID_U) break;
        }
        edge = next;
    } while (edge != start);
}

// ============================================================
// Downsample kernels (from QuadriFlow, adapted float/uint32_t)
// ============================================================

__global__ void k_score_entries(
    const float* N, const float* A,
    const uint32_t* adjRowPtr, const uint32_t* adjColInd,
    uint32_t* out_i, uint32_t* out_j, float* out_order, uint32_t nVerts)
{
    uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= nVerts) return;
    float nx,ny,nz; load_vec3f(N, v, nx,ny,nz);
    float av = A[v];
    uint32_t start = adjRowPtr[v], end = adjRowPtr[v+1];
    for (uint32_t idx = start; idx < end; ++idx) {
        uint32_t k = adjColInd[idx];
        float knx,kny,knz; load_vec3f(N, k, knx,kny,knz);
        float dp = nx*knx + ny*kny + nz*knz;
        float ak = A[k];
        float ratio = av > ak ? (av/ak) : (ak/av);
        out_i[idx] = v; out_j[idx] = k; out_order[idx] = dp * ratio;
    }
}

__global__ void __launch_bounds__(256) k_build_collapsed(
    const uint32_t* __restrict__ collapsed_i, const uint32_t* __restrict__ collapsed_j,
    const float* __restrict__ V, const float* __restrict__ N, const float* __restrict__ A,
    float* __restrict__ V_p, float* __restrict__ N_p, float* __restrict__ A_p,
    uint32_t* __restrict__ toUpper, uint32_t* __restrict__ toLower,
    uint32_t nCollapsed)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nCollapsed) return;
    uint32_t ei = collapsed_i[idx], ej = collapsed_j[idx];
    float a1 = A[ei], a2 = A[ej], sa = a1 + a2;
    float vix,viy,viz,vjx,vjy,vjz;
    load_vec3f(V, ei, vix,viy,viz); load_vec3f(V, ej, vjx,vjy,vjz);
    uint32_t b = idx*3;
    if (sa > RCPOVERFLOW_F) {
        float inv = 1.f/sa;
        V_p[b]=(vix*a1+vjx*a2)*inv; V_p[b+1]=(viy*a1+vjy*a2)*inv; V_p[b+2]=(viz*a1+vjz*a2)*inv;
    } else {
        V_p[b]=(vix+vjx)*.5f; V_p[b+1]=(viy+vjy)*.5f; V_p[b+2]=(viz+vjz)*.5f;
    }
    float nix,niy,niz,njx,njy,njz;
    load_vec3f(N, ei, nix,niy,niz); load_vec3f(N, ej, njx,njy,njz);
    float nnx=nix*a1+njx*a2, nny=niy*a1+njy*a2, nnz=niz*a1+njz*a2;
    float norm=sqrtf(nnx*nnx+nny*nny+nnz*nnz);
    if(norm>RCPOVERFLOW_F){float inv=1.f/norm;N_p[b]=nnx*inv;N_p[b+1]=nny*inv;N_p[b+2]=nnz*inv;}
    else{N_p[b]=1;N_p[b+1]=0;N_p[b+2]=0;}
    A_p[idx] = sa;
    toUpper[idx*2] = ei; toUpper[idx*2+1] = ej;
    toLower[ei] = idx; toLower[ej] = idx;
}

__global__ void k_copy_unmerged(
    const uint32_t* mergeFlag, const int* prefixSum,
    const float* V, const float* N, const float* A,
    float* V_p, float* N_p, float* A_p,
    uint32_t* toUpper, uint32_t* toLower,
    uint32_t nVerts, uint32_t nCollapsed)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nVerts || mergeFlag[i]) return;
    uint32_t idx = nCollapsed + (uint32_t)prefixSum[i];
    uint32_t sb=i*3, db=idx*3;
    V_p[db]=V[sb]; V_p[db+1]=V[sb+1]; V_p[db+2]=V[sb+2];
    N_p[db]=N[sb]; N_p[db+1]=N[sb+1]; N_p[db+2]=N[sb+2];
    A_p[idx] = A[i];
    toUpper[idx*2] = i; toUpper[idx*2+1] = INVALID_U;
    toLower[i] = idx;
}

__global__ void k_count_coarse_adj(
    const uint32_t* toUpper, const uint32_t* adjRowPtr,
    const uint32_t* toLower, uint32_t* counts, uint32_t nCoarse)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nCoarse) return;
    uint32_t total = 0;
    for (int j = 0; j < 2; ++j) {
        uint32_t upper = toUpper[j + i*2];
        if (upper == INVALID_U) continue;
        total += adjRowPtr[upper+1] - adjRowPtr[upper];
    }
    counts[i] = total;
}

__global__ void k_fill_coarse_adj(
    const uint32_t* toUpper, const uint32_t* adjRowPtr, const uint32_t* adjColInd,
    const float* adjWeights, const uint32_t* toLower,
    const uint32_t* outRowPtr, uint32_t* outColInd, float* outWeights,
    uint32_t* actualCount, uint32_t nCoarse)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nCoarse) return;
    uint32_t out_start = outRowPtr[i];
    int n = 0;
    for (int j = 0; j < 2; ++j) {
        uint32_t upper = toUpper[j + i*2];
        if (upper == INVALID_U) continue;
        uint32_t start = adjRowPtr[upper], end = adjRowPtr[upper+1];
        for (uint32_t k = start; k < end; ++k) {
            uint32_t mapped = toLower[adjColInd[k]];
            if (mapped != i) {
                outColInd[out_start+n] = mapped;
                outWeights[out_start+n] = adjWeights[k];
                n++;
            }
        }
    }
    // Sort + dedup (small n, typically <20)
    for (int a=1; a<n; ++a) {
        uint32_t kid=outColInd[out_start+a]; float kw=outWeights[out_start+a]; int b=a-1;
        while(b>=0 && outColInd[out_start+b]>kid) {
            outColInd[out_start+b+1]=outColInd[out_start+b];
            outWeights[out_start+b+1]=outWeights[out_start+b]; b--;
        }
        outColInd[out_start+b+1]=kid; outWeights[out_start+b+1]=kw;
    }
    int write=0;
    for (int a=0; a<n; ++a) {
        if(write>0 && outColInd[out_start+write-1]==outColInd[out_start+a])
            outWeights[out_start+write-1]+=outWeights[out_start+a];
        else { if(write!=a){outColInd[out_start+write]=outColInd[out_start+a];outWeights[out_start+write]=outWeights[out_start+a];} write++; }
    }
    actualCount[i] = write;
}

// ============================================================
// Random field init
// ============================================================

__global__ void k_random_tangent(const float* N, float* Q, uint32_t nVerts, uint32_t seed) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nVerts) return;
    float nx,ny,nz; load_vec3f(N, i, nx,ny,nz);
    // Build tangent frame
    float sx,sy,sz, cx,cy,cz;
    if (fabsf(nx) > fabsf(ny)) {
        float inv = 1.f/sqrtf(nx*nx+nz*nz);
        cx=nz*inv; cy=0; cz=-nx*inv;
    } else {
        float inv = 1.f/sqrtf(ny*ny+nz*nz);
        cx=0; cy=nz*inv; cz=-ny*inv;
    }
    sx=cy*nz-cz*ny; sy=cz*nx-cx*nz; sz=cx*ny-cy*nx;
    // Hash-based random angle
    uint32_t h = (i * 2654435761u + seed) ^ (i * 340573321u);
    float angle = (float)(h & 0xFFFFu) / 65535.f * 2.f * M_PI;
    float ca=cosf(angle), sa_=sinf(angle);
    uint32_t b=i*3;
    Q[b]=sx*ca+cx*sa_; Q[b+1]=sy*ca+cy*sa_; Q[b+2]=sz*ca+cz*sa_;
}

__global__ void k_random_position(const float* V, const float* N, float* O, float scale, uint32_t nVerts, uint32_t seed) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nVerts) return;
    float nx,ny,nz; load_vec3f(N, i, nx,ny,nz);
    float vx,vy,vz; load_vec3f(V, i, vx,vy,vz);
    float sx,sy,sz, cx,cy,cz;
    if (fabsf(nx) > fabsf(ny)) {
        float inv = 1.f/sqrtf(nx*nx+nz*nz);
        cx=nz*inv; cy=0; cz=-nx*inv;
    } else {
        float inv = 1.f/sqrtf(ny*ny+nz*nz);
        cx=0; cy=nz*inv; cz=-ny*inv;
    }
    sx=cy*nz-cz*ny; sy=cz*nx-cx*nz; sz=cx*ny-cy*nx;
    uint32_t h1 = (i * 2654435761u + seed) ^ (i * 340573321u);
    uint32_t h2 = (i * 1103515245u + seed + 1u) ^ (i * 12345u);
    float x = ((float)(h1 & 0xFFFFu) / 65535.f) * 2.f - 1.f;
    float y = ((float)(h2 & 0xFFFFu) / 65535.f) * 2.f - 1.f;
    uint32_t b=i*3;
    O[b]=vx+(sx*x+cx*y)*scale; O[b+1]=vy+(sy*x+cy*y)*scale; O[b+2]=vz+(sz*x+cz*y)*scale;
}

// ============================================================
// Optimizer kernels (reuse from optimizer_cuda.cu via extern declarations)
// We include the kernel declarations needed for the optimizer.
// The actual kernel code stays in optimizer_cuda.cu.
// ============================================================

// Forward declarations of optimizer kernels (defined in optimizer_cuda.cu)
extern __global__ void kernel_optimize_orientations(
    const uint32_t *phase_indices, uint32_t phase_size,
    const uint32_t *adj_row, const uint32_t *adj_col, const float *adj_weight,
    const float *N, const float *CQ, const float *CQw,
    float *Q, int orient_mode);

extern __global__ void kernel_optimize_positions(
    const uint32_t *phase_indices, uint32_t phase_size,
    const uint32_t *adj_row, const uint32_t *adj_col, const float *adj_weight,
    const float *V, const float *N, const float *Q_field,
    const float *CQ, const float *CO, const float *COw,
    float *O, float scale, float inv_scale, int pos_mode);

extern __global__ void kernel_propagate_orient(
    const float *src_Q, const uint32_t *toUpper, uint32_t nCoarse,
    float *dst_Q, const float *dst_N);

extern __global__ void kernel_propagate_pos(
    const float *src_O, const uint32_t *toUpper, uint32_t nCoarse,
    float *dst_O, const float *dst_N, const float *dst_V);

// ============================================================
// GPU Hierarchy struct
// ============================================================

struct GPULevel {
    float *d_V, *d_N, *d_A;
    uint32_t *d_adj_row, *d_adj_col;
    float *d_adj_weight;
    float *d_Q, *d_O;
    float *d_CQ, *d_CO, *d_CQw, *d_COw;
    uint32_t *d_toUpper;   // [nCoarse*2] maps this level's vertices to prev (finer) level
    uint32_t *d_toLower;   // [nFineVerts] during build only
    std::vector<uint32_t*> d_phases;
    std::vector<uint32_t> phase_sizes;
    uint32_t nVerts, nnz;
};

struct GPUHierarchy {
    std::vector<GPULevel> levels;
    uint32_t *d_F, *d_V2E, *d_E2E, *d_boundary, *d_nonManifold;
    float *d_Nf;
    unsigned int *d_tmp_next;
    uint32_t *d_tmp_opposite;
    uint32_t nFaces, nVerts0;
    int nLevels;
};

// ============================================================
// Implementation
// ============================================================

extern "C" {

GPUHierarchy* gpu_hierarchy_create() {
    GPUHierarchy* h = new GPUHierarchy();
    h->d_F = h->d_V2E = h->d_E2E = h->d_boundary = h->d_nonManifold = h->d_tmp_opposite = nullptr;
    h->d_Nf = nullptr; h->d_tmp_next = nullptr;
    h->nFaces = h->nVerts0 = 0; h->nLevels = 0;
    return h;
}

void gpu_hierarchy_destroy(GPUHierarchy* h) {
    if (!h) return;
    for (auto& lev : h->levels) {
        cudaFree(lev.d_V); cudaFree(lev.d_N); cudaFree(lev.d_A);
        cudaFree(lev.d_adj_row); cudaFree(lev.d_adj_col); cudaFree(lev.d_adj_weight);
        cudaFree(lev.d_Q); cudaFree(lev.d_O);
        if (lev.d_CQ) cudaFree(lev.d_CQ); if (lev.d_CO) cudaFree(lev.d_CO);
        if (lev.d_CQw) cudaFree(lev.d_CQw); if (lev.d_COw) cudaFree(lev.d_COw);
        if (lev.d_toUpper) cudaFree(lev.d_toUpper);
        if (lev.d_toLower) cudaFree(lev.d_toLower);
        for (auto p : lev.d_phases) cudaFree(p);
    }
    if (h->d_F) cudaFree(h->d_F);
    if (h->d_V2E) cudaFree(h->d_V2E);
    if (h->d_E2E) cudaFree(h->d_E2E);
    if (h->d_boundary) cudaFree(h->d_boundary);
    if (h->d_nonManifold) cudaFree(h->d_nonManifold);
    if (h->d_Nf) cudaFree(h->d_Nf);
    if (h->d_tmp_next) cudaFree(h->d_tmp_next);
    if (h->d_tmp_opposite) cudaFree(h->d_tmp_opposite);
    delete h;
}

void gpu_hierarchy_init(GPUHierarchy* h,
    const uint32_t* F, uint32_t nFaces, const float* V, uint32_t nVerts,
    uint32_t* h_V2E, uint32_t* h_E2E, uint32_t* h_boundary, uint32_t* h_nonManifold)
{
    h->nFaces = nFaces;
    h->nVerts0 = nVerts;
    uint32_t nEdges = 3 * nFaces;
    int gF = (nFaces+BS-1)/BS, gV = (nVerts+BS-1)/BS;

    // Allocate
    cudaMalloc(&h->d_F, nEdges * sizeof(uint32_t));
    cudaMalloc(&h->d_V2E, nVerts * sizeof(uint32_t));
    cudaMalloc(&h->d_E2E, nEdges * sizeof(uint32_t));
    cudaMalloc(&h->d_boundary, nVerts * sizeof(uint32_t));
    cudaMalloc(&h->d_nonManifold, nVerts * sizeof(uint32_t));
    cudaMalloc(&h->d_tmp_next, nEdges * sizeof(unsigned int));
    cudaMalloc(&h->d_tmp_opposite, nEdges * sizeof(uint32_t));
    cudaMalloc(&h->d_Nf, 3 * nFaces * sizeof(float));

    // Level 0
    h->levels.resize(1);
    GPULevel& L0 = h->levels[0];
    L0.nVerts = nVerts;
    cudaMalloc(&L0.d_V, 3*nVerts*sizeof(float));
    cudaMalloc(&L0.d_N, 3*nVerts*sizeof(float));
    cudaMalloc(&L0.d_A, nVerts*sizeof(float));
    cudaMalloc(&L0.d_Q, 3*nVerts*sizeof(float));
    cudaMalloc(&L0.d_O, 3*nVerts*sizeof(float));
    L0.d_CQ = L0.d_CO = L0.d_CQw = L0.d_COw = nullptr;
    L0.d_toUpper = L0.d_toLower = nullptr;

    // Upload F, V
    cudaMemcpy(h->d_F, F, nEdges*sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(L0.d_V, V, 3*nVerts*sizeof(float), cudaMemcpyHostToDevice);

    // Build dedge
    cudaMemset(h->d_V2E, 0xFF, nVerts*sizeof(uint32_t));
    cudaMemset(h->d_E2E, 0xFF, nEdges*sizeof(uint32_t));
    cudaMemset(h->d_tmp_next, 0xFF, nEdges*sizeof(unsigned int));
    cudaMemset(h->d_tmp_opposite, 0xFF, nEdges*sizeof(uint32_t));

    k_build_halfedge_lists<<<gF,BS>>>(h->d_F, nFaces, nVerts, h->d_V2E, h->d_tmp_next, h->d_tmp_opposite);
    cudaDeviceSynchronize();
    k_match_opposites<<<gF,BS>>>(h->d_F, nFaces, nVerts, h->d_V2E, h->d_tmp_next, h->d_tmp_opposite, h->d_E2E);
    cudaDeviceSynchronize();
    k_detect_boundary<<<gV,BS>>>(nVerts, h->d_V2E, h->d_E2E, h->d_boundary, h->d_nonManifold);
    cudaDeviceSynchronize();

    // Normals + areas
    k_face_normals<<<gF,BS>>>(h->d_F, L0.d_V, h->d_Nf, nFaces);
    k_smooth_normals<<<gV,BS>>>(h->d_F, L0.d_V, h->d_Nf, h->d_V2E, h->d_E2E, h->d_nonManifold, L0.d_N, nVerts);
    k_vertex_area<<<gV,BS>>>(h->d_F, L0.d_V, h->d_V2E, h->d_E2E, h->d_nonManifold, L0.d_A, nVerts);
    cudaDeviceSynchronize();

    // Build adjacency CSR on GPU
    uint32_t *d_counts;
    cudaMalloc(&d_counts, nVerts*sizeof(uint32_t));
    k_count_adj_from_dedge<<<gV,BS>>>(h->d_F, h->d_V2E, h->d_E2E, h->d_nonManifold, d_counts, nVerts);

    uint32_t *d_rowPtr;
    cudaMalloc(&d_rowPtr, (nVerts+1)*sizeof(uint32_t));
    thrust::device_ptr<uint32_t> cnt_ptr(d_counts), rp_ptr(d_rowPtr);
    thrust::exclusive_scan(cnt_ptr, cnt_ptr+nVerts, rp_ptr, (uint32_t)0);
    // Set last element
    uint32_t last_c, last_o;
    cudaMemcpy(&last_c, d_counts+nVerts-1, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&last_o, d_rowPtr+nVerts-1, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    uint32_t total_nnz = last_o + last_c;
    cudaMemcpy(d_rowPtr+nVerts, &total_nnz, sizeof(uint32_t), cudaMemcpyHostToDevice);

    uint32_t *d_colInd; float *d_weights;
    cudaMalloc(&d_colInd, total_nnz*sizeof(uint32_t));
    cudaMalloc(&d_weights, total_nnz*sizeof(float));
    k_fill_adj_from_dedge<<<gV,BS>>>(h->d_F, h->d_V2E, h->d_E2E, h->d_nonManifold, d_rowPtr, d_colInd, d_weights, nVerts);
    cudaDeviceSynchronize();

    L0.d_adj_row = d_rowPtr;
    L0.d_adj_col = d_colInd;
    L0.d_adj_weight = d_weights;
    L0.nnz = total_nnz;
    cudaFree(d_counts);

    // Check for errors from ALL init kernels (flush deferred errors)
    {
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
            fprintf(stderr, "[GPU HIER] init error after adj build: %s\n", cudaGetErrorString(err));
    }

    // Download V2E, E2E, boundary, nonManifold to CPU (needed for boundary alignment + extraction)
    cudaMemcpy(h_V2E, h->d_V2E, nVerts*sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_E2E, h->d_E2E, nEdges*sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_boundary, h->d_boundary, nVerts*sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_nonManifold, h->d_nonManifold, nVerts*sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Free temp buffers
    cudaFree(h->d_tmp_next); h->d_tmp_next = nullptr;
    cudaFree(h->d_tmp_opposite); h->d_tmp_opposite = nullptr;
    cudaFree(h->d_Nf); h->d_Nf = nullptr;

    h->nLevels = 1;
}

int gpu_hierarchy_build(GPUHierarchy* h, bool deterministic) {
    const int MAX_DEPTH = 25;
    h->levels.reserve(MAX_DEPTH + 1);  // prevent reallocation invalidating references

    for (int lvl = 0; lvl < MAX_DEPTH; ++lvl) {
        GPULevel& cur = h->levels[lvl];
        uint32_t nV = cur.nVerts;
        uint32_t nnz = cur.nnz;
        if (nV <= 1) break;

        int gV = (nV+BS-1)/BS;

        // Score entries on GPU
        uint32_t *d_ei, *d_ej; float *d_eo;
        cudaMalloc(&d_ei, nnz*sizeof(uint32_t));
        cudaMalloc(&d_ej, nnz*sizeof(uint32_t));
        cudaMalloc(&d_eo, nnz*sizeof(float));
        cudaFree(d_ei); cudaFree(d_ej); cudaFree(d_eo);

        // Score entries + sort + greedy merge all on CPU
        // Download N, A, adj for this level
        std::vector<float> h_N(nV*3), h_A(nV);
        std::vector<uint32_t> h_rp(nV+1), h_ci(nnz);
        cudaMemcpy(h_N.data(), cur.d_N, nV*3*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_A.data(), cur.d_A, nV*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_rp.data(), cur.d_adj_row, (nV+1)*sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_ci.data(), cur.d_adj_col, nnz*sizeof(uint32_t), cudaMemcpyDeviceToHost);

        // Score on CPU
        std::vector<uint32_t> h_ei(nnz), h_ej(nnz);
        std::vector<float> h_eo(nnz);
        for (uint32_t v = 0; v < nV; ++v) {
            float nx=h_N[v*3], ny=h_N[v*3+1], nz=h_N[v*3+2], av=h_A[v];
            for (uint32_t idx = h_rp[v]; idx < h_rp[v+1]; ++idx) {
                uint32_t k = h_ci[idx];
                float knx=h_N[k*3], kny=h_N[k*3+1], knz=h_N[k*3+2];
                float dp = nx*knx + ny*kny + nz*knz;
                float ak = h_A[k];
                float ratio = av > ak ? (av/ak) : (ak/av);
                h_ei[idx] = v; h_ej[idx] = k; h_eo[idx] = dp*ratio;
            }
        }

        // Sort (descending by order)
        std::vector<uint32_t> order(nnz);
        for (uint32_t i = 0; i < nnz; ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return h_eo[a] > h_eo[b]; });

        // Greedy merge
        std::vector<bool> merged(nV, false);
        std::vector<uint32_t> col_i, col_j;
        col_i.reserve(nV/2); col_j.reserve(nV/2);
        for (uint32_t k = 0; k < nnz; ++k) {
            uint32_t idx = order[k];
            uint32_t ei = h_ei[idx], ej = h_ej[idx];
            if (merged[ei] || merged[ej]) continue;
            merged[ei] = merged[ej] = true;
            col_i.push_back(ei); col_j.push_back(ej);
        }
        uint32_t nCollapsed = (uint32_t)col_i.size();
        uint32_t nCoarse = nV - nCollapsed;
        fprintf(stderr, "[GPU HIER] level %d: nCollapsed=%u nCoarse=%u\n", lvl, nCollapsed, nCoarse);
        // Validate vertex IDs
        for (uint32_t k=0; k<nCollapsed; ++k) {
            if (col_i[k] >= nV || col_j[k] >= nV) {
                fprintf(stderr, "[GPU HIER] BAD vertex ID at collapsed[%u]: i=%u j=%u (nV=%u)\n", k, col_i[k], col_j[k], nV);
                break;
            }
        }

        // Upload collapsed pairs + merge flags
        uint32_t *d_col_i, *d_col_j, *d_mflag;
        int *d_notm, *d_prefix;  // use int for thrust compatibility
        cudaMalloc(&d_col_i, nCollapsed*sizeof(uint32_t));
        cudaMalloc(&d_col_j, nCollapsed*sizeof(uint32_t));
        cudaMalloc(&d_mflag, nV*sizeof(uint32_t));
        cudaMalloc(&d_notm, nV*sizeof(int));
        cudaMalloc(&d_prefix, nV*sizeof(int));
        cudaMemcpy(d_col_i, col_i.data(), nCollapsed*sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_col_j, col_j.data(), nCollapsed*sizeof(uint32_t), cudaMemcpyHostToDevice);

        std::vector<uint32_t> mflag_h(nV);
        std::vector<int> notm_h(nV);
        for (uint32_t i=0; i<nV; ++i) { mflag_h[i]=merged[i]?1:0; notm_h[i]=merged[i]?0:1; }
        cudaMemcpy(d_mflag, mflag_h.data(), nV*sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_notm, notm_h.data(), nV*sizeof(int), cudaMemcpyHostToDevice);

        // New level
        h->levels.push_back(GPULevel());
        GPULevel& next = h->levels.back();
        next.nVerts = nCoarse;
        cudaMalloc(&next.d_V, 3*nCoarse*sizeof(float));
        cudaMalloc(&next.d_N, 3*nCoarse*sizeof(float));
        cudaMalloc(&next.d_A, nCoarse*sizeof(float));
        cudaMalloc(&next.d_Q, 3*nCoarse*sizeof(float));
        cudaMalloc(&next.d_O, 3*nCoarse*sizeof(float));
        next.d_CQ=next.d_CO=next.d_CQw=next.d_COw=nullptr;

        // toUpper for this level (maps coarse -> fine), stored at fine level
        cudaMalloc(&cur.d_toUpper, nCoarse*2*sizeof(uint32_t));

        // toLower for build (fine -> coarse)
        uint32_t *d_toLower;
        cudaMalloc(&d_toLower, nV*sizeof(uint32_t));

        // Build collapsed
        int gC = (nCollapsed+BS-1)/BS;
        fprintf(stderr, "[GPU HIER] ptrs: V=%p N=%p A=%p Vp=%p Np=%p Ap=%p toUpper=%p toLower=%p ci=%p cj=%p\n",
                cur.d_V, cur.d_N, cur.d_A, next.d_V, next.d_N, next.d_A, cur.d_toUpper, d_toLower, d_col_i, d_col_j);
        fprintf(stderr, "[GPU HIER] launching k_build_collapsed: gC=%d nCollapsed=%u\n", gC, nCollapsed);
        k_build_collapsed<<<gC,BS>>>(d_col_i, d_col_j, cur.d_V, cur.d_N, cur.d_A,
            next.d_V, next.d_N, next.d_A, cur.d_toUpper, d_toLower, nCollapsed);
        {
            cudaError_t e = cudaDeviceSynchronize();
            if (e != cudaSuccess) { fprintf(stderr, "[GPU HIER] k_build_collapsed error: %s\n", cudaGetErrorString(e)); break; }
        }

        // Prefix sum for unmerged
        thrust::device_ptr<int> notm_ptr(d_notm), pref_ptr(d_prefix);
        thrust::exclusive_scan(notm_ptr, notm_ptr+nV, pref_ptr, 0);

        // Copy unmerged
        k_copy_unmerged<<<gV,BS>>>(d_mflag, d_prefix, cur.d_V, cur.d_N, cur.d_A,
            next.d_V, next.d_N, next.d_A, cur.d_toUpper, d_toLower, nV, nCollapsed);
        cudaDeviceSynchronize();

        // Build coarse adjacency
        uint32_t *d_cnt2;
        cudaMalloc(&d_cnt2, nCoarse*sizeof(uint32_t));
        int gCoarse = (nCoarse+BS-1)/BS;
        k_count_coarse_adj<<<gCoarse,BS>>>(cur.d_toUpper, cur.d_adj_row, d_toLower, d_cnt2, nCoarse);

        uint32_t *d_rp2;
        cudaMalloc(&d_rp2, (nCoarse+1)*sizeof(uint32_t));
        thrust::device_ptr<uint32_t> cnt2_ptr(d_cnt2), rp2_ptr(d_rp2);
        thrust::exclusive_scan(cnt2_ptr, cnt2_ptr+nCoarse, rp2_ptr, (uint32_t)0);
        uint32_t lc2, lo2;
        cudaMemcpy(&lc2, d_cnt2+nCoarse-1, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&lo2, d_rp2+nCoarse-1, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        uint32_t nnz_upper = lo2 + lc2;
        cudaMemcpy(d_rp2+nCoarse, &nnz_upper, sizeof(uint32_t), cudaMemcpyHostToDevice);

        uint32_t *d_ci2, *d_ac2; float *d_w2;
        cudaMalloc(&d_ci2, nnz_upper*sizeof(uint32_t));
        cudaMalloc(&d_w2, nnz_upper*sizeof(float));
        cudaMalloc(&d_ac2, nCoarse*sizeof(uint32_t));

        k_fill_coarse_adj<<<gCoarse,BS>>>(cur.d_toUpper, cur.d_adj_row, cur.d_adj_col,
            cur.d_adj_weight, d_toLower, d_rp2, d_ci2, d_w2, d_ac2, nCoarse);
        cudaDeviceSynchronize();

        // For now, use the upper-bound CSR directly (optimizer doesn't care about gaps
        // since row_ptr correctly delimits each vertex's neighbors after dedup in k_fill_coarse_adj).
        // We need to compact using actualCount. Do it on CPU for simplicity.
        std::vector<uint32_t> c_rp(nCoarse+1), c_ci(nnz_upper), c_ac(nCoarse);
        std::vector<float> c_w(nnz_upper);
        cudaMemcpy(c_rp.data(), d_rp2, (nCoarse+1)*sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(c_ci.data(), d_ci2, nnz_upper*sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(c_w.data(), d_w2, nnz_upper*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(c_ac.data(), d_ac2, nCoarse*sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaFree(d_rp2); cudaFree(d_ci2); cudaFree(d_w2); cudaFree(d_ac2); cudaFree(d_cnt2);

        // Compact
        uint32_t actual_nnz = 0;
        for (uint32_t i=0; i<nCoarse; ++i) actual_nnz += c_ac[i];
        std::vector<uint32_t> compact_rp(nCoarse+1);
        std::vector<uint32_t> compact_ci(actual_nnz);
        std::vector<float> compact_w(actual_nnz);
        uint32_t wp = 0;
        compact_rp[0] = 0;
        for (uint32_t i=0; i<nCoarse; ++i) {
            uint32_t src = c_rp[i], cnt = c_ac[i];
            memcpy(&compact_ci[wp], &c_ci[src], cnt*sizeof(uint32_t));
            memcpy(&compact_w[wp], &c_w[src], cnt*sizeof(float));
            wp += cnt;
            compact_rp[i+1] = wp;
        }

        // Upload compacted adjacency
        cudaMalloc(&next.d_adj_row, (nCoarse+1)*sizeof(uint32_t));
        cudaMalloc(&next.d_adj_col, actual_nnz*sizeof(uint32_t));
        cudaMalloc(&next.d_adj_weight, actual_nnz*sizeof(float));
        cudaMemcpy(next.d_adj_row, compact_rp.data(), (nCoarse+1)*sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(next.d_adj_col, compact_ci.data(), actual_nnz*sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(next.d_adj_weight, compact_w.data(), actual_nnz*sizeof(float), cudaMemcpyHostToDevice);
        next.nnz = actual_nnz;
        next.d_toUpper = nullptr;
        next.d_toLower = nullptr;

        // Free build temps
        cudaFree(d_col_i); cudaFree(d_col_j);
        cudaFree(d_mflag); cudaFree(d_notm); cudaFree(d_prefix);
        cudaFree(d_toLower);

        h->nLevels = (int)h->levels.size();

        if (nCoarse <= 1) break;
    }

    return h->nLevels;
}

void gpu_hierarchy_init_fields(GPUHierarchy* h, float scale) {
    uint32_t seed = 42;
    for (int l = 0; l < h->nLevels; ++l) {
        GPULevel& lev = h->levels[l];
        int g = (lev.nVerts+BS-1)/BS;
        k_random_tangent<<<g,BS>>>(lev.d_N, lev.d_Q, lev.nVerts, seed + l*1000);
        k_random_position<<<g,BS>>>(lev.d_V, lev.d_N, lev.d_O, scale, lev.nVerts, seed + l*1000 + 500);
    }
    cudaDeviceSynchronize();
}

void gpu_hierarchy_upload_constraints(GPUHierarchy* h,
    int level, const float* CQ, const float* CO,
    const float* CQw, const float* COw, uint32_t nVerts)
{
    GPULevel& lev = h->levels[level];
    if (!lev.d_CQ) cudaMalloc(&lev.d_CQ, 3*nVerts*sizeof(float));
    if (!lev.d_CO) cudaMalloc(&lev.d_CO, 3*nVerts*sizeof(float));
    if (!lev.d_CQw) cudaMalloc(&lev.d_CQw, nVerts*sizeof(float));
    if (!lev.d_COw) cudaMalloc(&lev.d_COw, nVerts*sizeof(float));
    cudaMemcpy(lev.d_CQ, CQ, 3*nVerts*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(lev.d_CO, CO, 3*nVerts*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(lev.d_CQw, CQw, nVerts*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(lev.d_COw, COw, nVerts*sizeof(float), cudaMemcpyHostToDevice);
}

void gpu_hierarchy_upload_phases(GPUHierarchy* h, int level,
    const uint32_t* const* phase_ptrs, const uint32_t* phase_sizes_arr, uint32_t nPhases)
{
    GPULevel& lev = h->levels[level];
    lev.d_phases.resize(nPhases);
    lev.phase_sizes.resize(nPhases);
    for (uint32_t p = 0; p < nPhases; ++p) {
        lev.phase_sizes[p] = phase_sizes_arr[p];
        cudaMalloc(&lev.d_phases[p], phase_sizes_arr[p]*sizeof(uint32_t));
        cudaMemcpy(lev.d_phases[p], phase_ptrs[p], phase_sizes_arr[p]*sizeof(uint32_t), cudaMemcpyHostToDevice);
    }
}

// GPU-resident optimize stubs (experimental path, strategy==2)
// These use the phase-based kernels from optimizer_cuda.cu via the extern declarations above.
// Currently not called in the default pipeline — the hybrid path in batch.cpp uses
// cuda_run_optimization() instead.
void gpu_hierarchy_optimize_orient(GPUHierarchy* h,
    int nLevels, int rosy, int posy, bool extrinsic, float scale)
{
    int orient_mode = rosy * 2 + (extrinsic ? 0 : 1);
    const int levelIterations = 6;
    for (int level = nLevels-1; level >= 0; --level) {
        GPULevel& lev = h->levels[level];
        for (int iter = 0; iter < levelIterations; ++iter) {
            for (uint32_t p = 0; p < lev.d_phases.size(); ++p) {
                uint32_t ps = lev.phase_sizes[p];
                if (ps == 0) continue;
                int grid = (ps+BS-1)/BS;
                kernel_optimize_orientations<<<grid,BS>>>(
                    lev.d_phases[p], ps, lev.d_adj_row, lev.d_adj_col, lev.d_adj_weight,
                    lev.d_N, lev.d_CQ, lev.d_CQw, lev.d_Q, orient_mode);
            }
            cudaDeviceSynchronize();
        }
        if (level > 0) {
            GPULevel& fine = h->levels[level-1];
            if (fine.d_toUpper) {
                int pg = (lev.nVerts+BS-1)/BS;
                kernel_propagate_orient<<<pg,BS>>>(lev.d_Q, fine.d_toUpper, lev.nVerts, fine.d_Q, fine.d_N);
                cudaDeviceSynchronize();
            }
        }
    }
}

void gpu_hierarchy_optimize_position(GPUHierarchy* h,
    int nLevels, int rosy, int posy, bool extrinsic, float scale)
{
    int pos_mode = posy * 2 + (extrinsic ? 0 : 1);
    float inv_scale = 1.0f / scale;
    const int levelIterations = 6;
    for (int level = nLevels-1; level >= 0; --level) {
        GPULevel& lev = h->levels[level];
        for (int iter = 0; iter < levelIterations; ++iter) {
            for (uint32_t p = 0; p < lev.d_phases.size(); ++p) {
                uint32_t ps = lev.phase_sizes[p];
                if (ps == 0) continue;
                int grid = (ps+BS-1)/BS;
                kernel_optimize_positions<<<grid,BS>>>(
                    lev.d_phases[p], ps, lev.d_adj_row, lev.d_adj_col, lev.d_adj_weight,
                    lev.d_V, lev.d_N, lev.d_Q, lev.d_CQ, lev.d_CO, lev.d_COw,
                    lev.d_O, scale, inv_scale, pos_mode);
            }
            cudaDeviceSynchronize();
        }
        if (level > 0) {
            GPULevel& fine = h->levels[level-1];
            if (fine.d_toUpper) {
                int pg = (lev.nVerts+BS-1)/BS;
                kernel_propagate_pos<<<pg,BS>>>(lev.d_O, fine.d_toUpper, lev.nVerts, fine.d_O, fine.d_N, fine.d_V);
                cudaDeviceSynchronize();
            }
        }
    }
}

// Download functions
void gpu_hierarchy_download_Q(GPUHierarchy* h, int level, float* Q, uint32_t nVerts) {
    cudaMemcpy(Q, h->levels[level].d_Q, 3*nVerts*sizeof(float), cudaMemcpyDeviceToHost);
}
void gpu_hierarchy_download_O(GPUHierarchy* h, int level, float* O, uint32_t nVerts) {
    cudaMemcpy(O, h->levels[level].d_O, 3*nVerts*sizeof(float), cudaMemcpyDeviceToHost);
}
void gpu_hierarchy_download_V(GPUHierarchy* h, int level, float* V, uint32_t nVerts) {
    cudaMemcpy(V, h->levels[level].d_V, 3*nVerts*sizeof(float), cudaMemcpyDeviceToHost);
}
void gpu_hierarchy_download_N(GPUHierarchy* h, int level, float* N, uint32_t nVerts) {
    cudaMemcpy(N, h->levels[level].d_N, 3*nVerts*sizeof(float), cudaMemcpyDeviceToHost);
}
void gpu_hierarchy_download_A(GPUHierarchy* h, int level, float* A, uint32_t nVerts) {
    cudaMemcpy(A, h->levels[level].d_A, nVerts*sizeof(float), cudaMemcpyDeviceToHost);
}
int gpu_hierarchy_num_levels(GPUHierarchy* h) { return h->nLevels; }
uint32_t gpu_hierarchy_level_nVerts(GPUHierarchy* h, int level) { return h->levels[level].nVerts; }

uint32_t gpu_hierarchy_download_adj(GPUHierarchy* h, int level,
    uint32_t* row_ptr, uint32_t* col_idx, float* weights)
{
    GPULevel& lev = h->levels[level];
    if (row_ptr)
        cudaMemcpy(row_ptr, lev.d_adj_row, (lev.nVerts+1)*sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (col_idx)
        cudaMemcpy(col_idx, lev.d_adj_col, lev.nnz*sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (weights)
        cudaMemcpy(weights, lev.d_adj_weight, lev.nnz*sizeof(float), cudaMemcpyDeviceToHost);
    return lev.nnz;
}

void gpu_hierarchy_download_toUpper(GPUHierarchy* h, int level, uint32_t* toUpper, uint32_t nCoarseVerts) {
    if (h->levels[level].d_toUpper)
        cudaMemcpy(toUpper, h->levels[level].d_toUpper, nCoarseVerts*2*sizeof(uint32_t), cudaMemcpyDeviceToHost);
}

void gpu_hierarchy_download_toLower(GPUHierarchy* h, int level, uint32_t* toLower, uint32_t nFineVerts) {
    if (h->levels[level].d_toLower)
        cudaMemcpy(toLower, h->levels[level].d_toLower, nFineVerts*sizeof(uint32_t), cudaMemcpyDeviceToHost);
}

} // extern "C"
