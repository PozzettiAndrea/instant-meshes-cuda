// ============================================================
// GPU-resident parallel mesh subdivision for Instant Meshes
//
// Ported from QuadriFlow-cuda. Key changes:
//   - float (not double) — Instant Meshes uses single precision
//   - uint32_t indices, INVALID = 0xFFFFFFFF
//   - No rho sizing field — just maxLength threshold
//
// Single H->D upload, all passes on GPU, single D->H download.
// Each pass:
//   1. GPU marks long edges
//   2. GPU resolves within-face conflicts (keep longest)
//   3. GPU resolves cross-face conflicts (lower index wins)
//   4. GPU resolves neighbor conflicts (Luby-like independent set)
//   5. thrust::reduce to count splits. If 0, done.
//   6. thrust::exclusive_scan for vertex/face output offsets
//   7. GPU applies all splits in parallel (double-buffered F)
//   8. GPU rebuilds E2E via sort-based pairing
// ============================================================

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <algorithm>

#include "subdivide_gpu.h"

static constexpr uint32_t INVALID = 0xFFFFFFFF;

// ---- Kernel: mark edges that are too long ----
__global__ void k_mark_long_edges(
    const float* V,            // [3 x nV] col-major
    const uint32_t* F,         // [3 x nF] col-major
    const uint32_t* E2E,       // [3*nF]
    const uint32_t* nonmanifold, // [nV]
    uint32_t nF,
    float maxLengthSq,
    int* edge_marks            // [3*nF] output
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nF * 3) return;

    uint32_t f = idx / 3;
    uint32_t j = idx % 3;
    uint32_t v0 = F[j + f * 3];
    uint32_t v1 = F[((j + 1) % 3) + f * 3];

    edge_marks[idx] = 0;
    if (nonmanifold[v0] || nonmanifold[v1]) return;

    // Canonical: only mark if boundary or idx < twin
    uint32_t other = E2E[idx];
    if (other != INVALID && other < idx) return;

    float d0 = V[0 + v0 * 3] - V[0 + v1 * 3];
    float d1 = V[1 + v0 * 3] - V[1 + v1 * 3];
    float d2 = V[2 + v0 * 3] - V[2 + v1 * 3];
    float lengthSq = d0 * d0 + d1 * d1 + d2 * d2;

    if (lengthSq > maxLengthSq) {
        edge_marks[idx] = 1;
    }
}

// ---- Kernel: resolve within-face conflicts (keep longest) ----
__global__ void k_resolve_conflicts(
    const float* V,
    const uint32_t* F,
    uint32_t nF,
    int* edge_marks
) {
    uint32_t f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nF) return;

    int count = edge_marks[f * 3 + 0] + edge_marks[f * 3 + 1] + edge_marks[f * 3 + 2];
    if (count <= 1) return;

    float best_len = -1.0f;
    int best_j = -1;
    for (int j = 0; j < 3; ++j) {
        if (!edge_marks[f * 3 + j]) continue;
        uint32_t v0 = F[j + f * 3];
        uint32_t v1 = F[((j + 1) % 3) + f * 3];
        float d0 = V[0 + v0 * 3] - V[0 + v1 * 3];
        float d1 = V[1 + v0 * 3] - V[1 + v1 * 3];
        float d2 = V[2 + v0 * 3] - V[2 + v1 * 3];
        float len = d0 * d0 + d1 * d1 + d2 * d2;
        if (len > best_len) {
            best_len = len;
            best_j = j;
        }
    }
    for (int j = 0; j < 3; ++j) {
        if (j != best_j) edge_marks[f * 3 + j] = 0;
    }
}

// ---- Kernel: resolve cross-face conflicts (lower index wins) ----
__global__ void k_resolve_cross_face(
    const uint32_t* E2E,
    uint32_t nF,
    int* edge_marks
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nF * 3) return;
    if (!edge_marks[idx]) return;

    uint32_t other = E2E[idx];
    if (other != INVALID && edge_marks[other] && idx > other) {
        edge_marks[idx] = 0;
    }
}

// ---- Kernel: resolve neighbor conflicts (Luby-like independent set) ----
// A split of edge (idx) rewrites both f0 (containing idx) and f1 (twin's face).
// Two marks that share a face must not both survive.
// Longest-first priority produces near-optimal triangulations matching CPU.
__global__ void k_resolve_neighbor_conflicts(
    const float* V,
    const uint32_t* F,
    const uint32_t* E2E,
    uint32_t nE,
    int* edge_marks
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nE) return;
    if (!edge_marks[idx]) return;

    // Compute my edge length squared
    uint32_t f0 = idx / 3, j0 = idx % 3;
    uint32_t va = F[j0 + f0 * 3], vb = F[((j0 + 1) % 3) + f0 * 3];
    float dx = V[va * 3] - V[vb * 3];
    float dy = V[va * 3 + 1] - V[vb * 3 + 1];
    float dz = V[va * 3 + 2] - V[vb * 3 + 2];
    float my_len = dx * dx + dy * dy + dz * dz;

    // Check if another marked edge beats us (longer wins, higher index breaks ties)
    auto dominated_by = [&](uint32_t other_idx) -> bool {
        if (other_idx == INVALID || !edge_marks[other_idx]) return false;
        uint32_t fo = other_idx / 3, jo = other_idx % 3;
        uint32_t ua = F[jo + fo * 3], ub = F[((jo + 1) % 3) + fo * 3];
        float ex = V[ua * 3] - V[ub * 3];
        float ey = V[ua * 3 + 1] - V[ub * 3 + 1];
        float ez = V[ua * 3 + 2] - V[ub * 3 + 2];
        float other_len = ex * ex + ey * ey + ez * ez;
        return (other_len > my_len) || (other_len == my_len && other_idx > idx);
    };

    // Check f0 (our face): do any of our face-mates' twins bring in a longer mark?
    for (uint32_t j = 0; j < 3; j++) {
        uint32_t he = f0 * 3 + j;
        if (he == idx) continue;
        uint32_t twin = E2E[he];
        if (dominated_by(twin)) { edge_marks[idx] = 0; return; }
    }

    // Check f1 (twin's face)
    uint32_t twin_of_idx = E2E[idx];
    if (twin_of_idx == INVALID) return;  // boundary, no f1
    uint32_t f1 = twin_of_idx / 3;
    for (uint32_t j = 0; j < 3; j++) {
        uint32_t he = f1 * 3 + j;
        if (he == twin_of_idx) continue;  // skip twin (same edge as us)
        if (dominated_by(he)) { edge_marks[idx] = 0; return; }
        uint32_t twin2 = E2E[he];
        if (dominated_by(twin2)) { edge_marks[idx] = 0; return; }
    }
}

// ---- Kernel: compute face counts per marked edge ----
// For each marked half-edge: 1 new face if boundary, 2 if interior
__global__ void k_compute_face_counts(
    const int* edge_marks,
    const uint32_t* E2E,
    uint32_t nE,
    int* face_counts  // [nE] output
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nE) return;
    if (!edge_marks[idx]) { face_counts[idx] = 0; return; }
    face_counts[idx] = (E2E[idx] == INVALID) ? 1 : 2;
}

// ---- Kernel: apply splits in parallel ----
// Reads from F_old (snapshot), writes to F, V, nm, bnd
__global__ void k_apply_splits(
    const uint32_t* F_old,      // snapshot of F before this pass
    uint32_t* F,                // modified in-place
    float* V,
    uint32_t* nm,
    uint32_t* bnd,
    const uint32_t* E2E,        // E2E from before this pass
    const int* edge_marks,
    const int* vtx_scan,        // exclusive scan of marks
    const int* face_scan,       // exclusive scan of face_counts
    uint32_t nV_old,
    uint32_t nF_old,
    uint32_t nE_old
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nE_old) return;
    if (!edge_marks[idx]) return;

    uint32_t f0 = idx / 3, j0 = idx % 3;
    uint32_t e1 = E2E[idx];
    bool is_boundary = (e1 == INVALID);

    // Read from snapshot
    uint32_t v0  = F_old[j0 + f0 * 3];
    uint32_t v0p = F_old[((j0 + 2) % 3) + f0 * 3];
    uint32_t v1  = F_old[((j0 + 1) % 3) + f0 * 3];

    uint32_t vn = nV_old + (uint32_t)vtx_scan[idx];

    // New vertex = midpoint
    V[0 + vn * 3] = 0.5f * (V[0 + v0 * 3] + V[0 + v1 * 3]);
    V[1 + vn * 3] = 0.5f * (V[1 + v0 * 3] + V[1 + v1 * 3]);
    V[2 + vn * 3] = 0.5f * (V[2 + v0 * 3] + V[2 + v1 * 3]);
    nm[vn] = 0;
    bnd[vn] = is_boundary ? 1 : 0;

    // Rewrite f0: (vn, v0p, v0)
    F[0 + f0 * 3] = vn;
    F[1 + f0 * 3] = v0p;
    F[2 + f0 * 3] = v0;

    // New face f3: (vn, v1, v0p)
    uint32_t f3 = nF_old + (uint32_t)face_scan[idx];
    F[0 + f3 * 3] = vn;
    F[1 + f3 * 3] = v1;
    F[2 + f3 * 3] = v0p;

    if (!is_boundary) {
        uint32_t f1 = e1 / 3, j1 = e1 % 3;
        uint32_t v1p = F_old[((j1 + 2) % 3) + f1 * 3];

        // Rewrite f1: (vn, v0, v1p)
        F[0 + f1 * 3] = vn;
        F[1 + f1 * 3] = v0;
        F[2 + f1 * 3] = v1p;

        // New face f2: (vn, v1p, v1)
        uint32_t f2 = nF_old + (uint32_t)face_scan[idx] + 1;
        F[0 + f2 * 3] = vn;
        F[1 + f2 * 3] = v1p;
        F[2 + f2 * 3] = v1;
    }
}

// ---- Kernel: build sort keys for E2E reconstruction ----
// key = min(va,vb) * maxV + max(va,vb), value = half-edge index
__global__ void k_build_e2e_keys(
    const uint32_t* F,
    uint32_t nF,
    uint32_t maxV,              // upper bound on vertex index (for key packing)
    long long* keys,            // [3*nF] output
    int* indices                // [3*nF] output
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nF * 3) return;

    uint32_t f = idx / 3, j = idx % 3;
    uint32_t va = F[j + f * 3];
    uint32_t vb = F[((j + 1) % 3) + f * 3];
    long long lo = (va < vb) ? va : vb;
    long long hi = (va < vb) ? vb : va;
    keys[idx] = lo * (long long)maxV + hi;
    indices[idx] = (int)idx;
}

// ---- Kernel: pair sorted half-edges to build E2E ----
// After sorting by key, consecutive entries with same key are twins.
__global__ void k_pair_e2e(
    const long long* sorted_keys,
    const int* sorted_indices,
    uint32_t nE,
    uint32_t* E2E   // pre-filled with INVALID
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nE) return;

    long long my_key = sorted_keys[i];
    bool is_first = (i == 0 || sorted_keys[i - 1] != my_key);
    bool has_next = (i + 1 < nE && sorted_keys[i + 1] == my_key);

    if (is_first && has_next) {
        // This is the first of a twin pair — link them
        uint32_t h0 = (uint32_t)sorted_indices[i];
        uint32_t h1 = (uint32_t)sorted_indices[i + 1];
        E2E[h0] = h1;
        E2E[h1] = h0;
    }
    // Boundary edges (unpaired) keep E2E = INVALID from the fill
}

// ============================================================
// Host function: GPU-resident mesh subdivision
// ============================================================
extern "C"
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
    uint32_t** nonmanifold_out
) {
    float maxLengthSq = maxLength * maxLength;
    uint32_t nV = nV_in, nF = nF_in;

    // Capacity: 10x initial size, minimum 1000
    uint32_t capV = std::max(nV_in * 10, (uint32_t)1000);
    uint32_t capF = std::max(nF_in * 10, (uint32_t)2000);
    uint32_t capE = capF * 3;

    // Allocate GPU arrays with capacity
    float *d_V;
    uint32_t *d_F, *d_E2E, *d_nm, *d_bnd;

    cudaMalloc(&d_V,   3 * capV * sizeof(float));
    cudaMalloc(&d_F,   3 * capF * sizeof(uint32_t));
    cudaMalloc(&d_E2E, capE * sizeof(uint32_t));
    cudaMalloc(&d_nm,  capV * sizeof(uint32_t));
    cudaMalloc(&d_bnd, capV * sizeof(uint32_t));

    // Work arrays (allocated at max capacity)
    int *d_marks, *d_face_counts, *d_vtx_scan, *d_face_scan;
    uint32_t *d_F_old;
    long long *d_sort_keys;
    int *d_sort_indices;

    cudaMalloc(&d_marks,        capE * sizeof(int));
    cudaMalloc(&d_face_counts,  capE * sizeof(int));
    cudaMalloc(&d_vtx_scan,     capE * sizeof(int));
    cudaMalloc(&d_face_scan,    capE * sizeof(int));
    cudaMalloc(&d_F_old,        3 * capF * sizeof(uint32_t));
    cudaMalloc(&d_sort_keys,    capE * sizeof(long long));
    cudaMalloc(&d_sort_indices, capE * sizeof(int));

    // Upload once
    cudaMemcpy(d_V,   V_in,          3 * nV * sizeof(float),    cudaMemcpyHostToDevice);
    cudaMemcpy(d_F,   F_in,          3 * nF * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_E2E, E2E_in,        3 * nF * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nm,  nonmanifold_in, nV * sizeof(uint32_t),    cudaMemcpyHostToDevice);
    cudaMemcpy(d_bnd, boundary_in,    nV * sizeof(uint32_t),    cudaMemcpyHostToDevice);

    int total_splits = 0, pass = 0;
    const int block = 256;

    while (true) {
        uint32_t nE = nF * 3;
        int grid_e = ((int)nE + block - 1) / block;
        int grid_f = ((int)nF + block - 1) / block;

        // 1-4. Mark + resolve conflicts
        k_mark_long_edges<<<grid_e, block>>>(d_V, d_F, d_E2E, d_nm, nF, maxLengthSq, d_marks);
        k_resolve_conflicts<<<grid_f, block>>>(d_V, d_F, nF, d_marks);
        k_resolve_cross_face<<<grid_e, block>>>(d_E2E, nF, d_marks);
        k_resolve_neighbor_conflicts<<<grid_e, block>>>(d_V, d_F, d_E2E, nE, d_marks);

        // Count splits
        thrust::device_ptr<int> marks_ptr(d_marks);
        int num_splits = thrust::reduce(marks_ptr, marks_ptr + nE);
        if (num_splits == 0) break;
        total_splits += num_splits;

        // Exclusive scan of marks -> vertex offsets
        thrust::device_ptr<int> vtx_scan_ptr(d_vtx_scan);
        thrust::exclusive_scan(marks_ptr, marks_ptr + nE, vtx_scan_ptr);

        // Compute face counts and scan
        k_compute_face_counts<<<grid_e, block>>>(d_marks, d_E2E, nE, d_face_counts);
        thrust::device_ptr<int> fc_ptr(d_face_counts);
        thrust::device_ptr<int> face_scan_ptr(d_face_scan);
        thrust::exclusive_scan(fc_ptr, fc_ptr + nE, face_scan_ptr);

        // Get total new faces from last element
        int last_fc, last_fs;
        cudaMemcpy(&last_fc, d_face_counts + nE - 1, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&last_fs, d_face_scan + nE - 1,   sizeof(int), cudaMemcpyDeviceToHost);
        int total_new_faces = last_fs + last_fc;

        uint32_t new_nV = nV + (uint32_t)num_splits;
        uint32_t new_nF = nF + (uint32_t)total_new_faces;

        // Check capacity
        if (new_nV > capV || new_nF > capF) {
            fprintf(stderr, "[SUBDIV-GPU] ERROR: capacity exceeded! nV=%u/%u nF=%u/%u\n",
                    new_nV, capV, new_nF, capF);
            break;
        }

        // Snapshot F
        cudaMemcpy(d_F_old, d_F, 3 * nF * sizeof(uint32_t), cudaMemcpyDeviceToDevice);

        // Apply splits
        k_apply_splits<<<grid_e, block>>>(
            d_F_old, d_F, d_V, d_nm, d_bnd,
            d_E2E, d_marks, d_vtx_scan, d_face_scan,
            nV, nF, nE);

        // Update counts
        nV = new_nV;
        nF = new_nF;
        uint32_t new_nE = nF * 3;
        int new_grid_e = ((int)new_nE + block - 1) / block;

        // Build E2E sort keys
        k_build_e2e_keys<<<new_grid_e, block>>>(d_F, nF, nV, d_sort_keys, d_sort_indices);

        // Sort by key
        thrust::device_ptr<long long> keys_ptr(d_sort_keys);
        thrust::device_ptr<int> idx_ptr(d_sort_indices);
        thrust::sort_by_key(keys_ptr, keys_ptr + new_nE, idx_ptr);

        // Initialize E2E to INVALID, then pair
        thrust::device_ptr<uint32_t> e2e_ptr(d_E2E);
        thrust::fill(e2e_ptr, e2e_ptr + new_nE, INVALID);

        k_pair_e2e<<<new_grid_e, block>>>(d_sort_keys, d_sort_indices, new_nE, d_E2E);

        pass++;
    }

    // Download once
    *nV_out = nV;
    *nF_out = nF;
    *V_out           = (float*)malloc(3 * nV * sizeof(float));
    *F_out           = (uint32_t*)malloc(3 * nF * sizeof(uint32_t));
    *boundary_out    = (uint32_t*)malloc(nV * sizeof(uint32_t));
    *nonmanifold_out = (uint32_t*)malloc(nV * sizeof(uint32_t));

    cudaMemcpy(*V_out,           d_V,   3 * nV * sizeof(float),    cudaMemcpyDeviceToHost);
    cudaMemcpy(*F_out,           d_F,   3 * nF * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(*boundary_out,    d_bnd, nV * sizeof(uint32_t),     cudaMemcpyDeviceToHost);
    cudaMemcpy(*nonmanifold_out, d_nm,  nV * sizeof(uint32_t),     cudaMemcpyDeviceToHost);

    // Free all GPU memory
    cudaFree(d_V); cudaFree(d_F); cudaFree(d_E2E);
    cudaFree(d_nm); cudaFree(d_bnd);
    cudaFree(d_marks); cudaFree(d_face_counts);
    cudaFree(d_vtx_scan); cudaFree(d_face_scan);
    cudaFree(d_F_old);
    cudaFree(d_sort_keys); cudaFree(d_sort_indices);
}
