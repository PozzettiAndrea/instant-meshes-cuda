/*
    optimizer_cuda.cu -- CUDA kernels for field optimization in Instant Meshes

    GPU-accelerated orientation and position field smoothing across the
    multi-resolution hierarchy. Converts the pointer-based AdjacencyMatrix
    to flat CSR arrays, uploads once, runs multiple kernel passes per level,
    downloads once.

    Supports rosy={2,4,6} x extrinsic/intrinsic for orientations
    and posy={3,4} x extrinsic/intrinsic for positions.
*/

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>

// ============================================================
// Device math helpers (float3 operations)
// ============================================================

__device__ __forceinline__ float3 make_f3(float x, float y, float z) {
    return make_float3(x, y, z);
}

__device__ __forceinline__ float3 operator+(float3 a, float3 b) {
    return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}

__device__ __forceinline__ float3 operator-(float3 a, float3 b) {
    return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}

__device__ __forceinline__ float3 operator*(float3 a, float s) {
    return make_float3(a.x*s, a.y*s, a.z*s);
}

__device__ __forceinline__ float3 operator*(float s, float3 a) {
    return make_float3(a.x*s, a.y*s, a.z*s);
}

__device__ __forceinline__ float dot3(float3 a, float3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

__device__ __forceinline__ float3 cross3(float3 a, float3 b) {
    return make_float3(a.y*b.z - a.z*b.y,
                       a.z*b.x - a.x*b.z,
                       a.x*b.y - a.y*b.x);
}

__device__ __forceinline__ float norm3(float3 a) {
    return sqrtf(dot3(a, a));
}

__device__ __forceinline__ float3 normalize3(float3 a) {
    float n = norm3(a);
    if (n > 1e-38f) return a * (1.0f / n);
    return a;
}

__device__ __forceinline__ float d_signum(float v) {
    return copysignf(1.0f, v);
}

__device__ __forceinline__ float d_fast_acos(float x) {
    float negate = float(x < 0.0f);
    x = fabsf(x);
    float ret = -0.0187293f;
    ret *= x; ret = ret + 0.0742610f;
    ret *= x; ret = ret - 0.2121144f;
    ret *= x; ret = ret + 1.5707288f;
    ret = ret * sqrtf(1.0f - x);
    ret = ret - 2.0f * negate * ret;
    return negate * 3.14159265358979323846f + ret;
}

static const float SQRT_3_OVER_4 = 0.866025403784439f;

__device__ __forceinline__ float3 d_rotate60(float3 d, float3 n) {
    return SQRT_3_OVER_4 * cross3(n, d) + 0.5f * (d + n * dot3(n, d));
}

__device__ __forceinline__ float3 d_rotate90(float3 q, float3 n) {
    return cross3(n, q);
}

// ============================================================
// Device compat functions for orientation (extrinsic)
// ============================================================

// Extrinsic 4-RoSy: find best matching rotation of q1 around n1 to q0
__device__ void d_compat_orient_extrinsic_4(
    float3 q0, float3 n0, float3 q1, float3 n1,
    float3 &out0, float3 &out1)
{
    float3 A[2] = { q0, cross3(n0, q0) };
    float3 B[2] = { q1, cross3(n1, q1) };

    float best_score = -1e30f;
    int best_a = 0, best_b = 0;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            float score = fabsf(dot3(A[i], B[j]));
            if (score > best_score) {
                best_a = i; best_b = j;
                best_score = score;
            }
        }
    }
    float dp = dot3(A[best_a], B[best_b]);
    out0 = A[best_a];
    out1 = B[best_b] * d_signum(dp);
}

// Extrinsic 2-RoSy
__device__ void d_compat_orient_extrinsic_2(
    float3 q0, float3 n0, float3 q1, float3 n1,
    float3 &out0, float3 &out1)
{
    out0 = q0;
    out1 = q1 * d_signum(dot3(q0, q1));
}

// Extrinsic 6-RoSy
__device__ void d_compat_orient_extrinsic_6(
    float3 q0, float3 n0, float3 q1, float3 n1,
    float3 &out0, float3 &out1)
{
    float3 n0_neg = make_f3(-n0.x, -n0.y, -n0.z);
    float3 n1_neg = make_f3(-n1.x, -n1.y, -n1.z);
    float3 A[3] = { d_rotate60(q0, n0_neg), q0, d_rotate60(q0, n0) };
    float3 B[3] = { d_rotate60(q1, n1_neg), q1, d_rotate60(q1, n1) };

    float best_score = -1e30f;
    int best_a = 0, best_b = 0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            float score = fabsf(dot3(A[i], B[j]));
            if (score > best_score) {
                best_a = i; best_b = j;
                best_score = score;
            }
        }
    }
    float dp = dot3(A[best_a], B[best_b]);
    out0 = A[best_a];
    out1 = B[best_b] * d_signum(dp);
}

// Intrinsic helpers
__device__ float3 d_rotate_vector_into_plane(float3 q, float3 source_normal, float3 target_normal) {
    float3 axis = cross3(source_normal, target_normal);
    float cosTheta = dot3(source_normal, target_normal);
    if (cosTheta < 0.9999f) {
        float factor = (1.0f - cosTheta) / (dot3(axis, axis) + 1e-10f);
        q = q * cosTheta + cross3(axis, q) + axis * (dot3(axis, q) * factor);
    }
    return q;
}

__device__ void d_compat_orient_intrinsic_4(
    float3 q0, float3 n0, float3 q1, float3 n1,
    float3 &out0, float3 &out1)
{
    float3 q1r = d_rotate_vector_into_plane(q1, n1, n0);
    float3 t1 = cross3(n0, q1r);
    float dp0 = dot3(q1r, q0), dp1 = dot3(t1, q0);
    out0 = q0;
    if (fabsf(dp0) > fabsf(dp1))
        out1 = q1r * d_signum(dp0);
    else
        out1 = t1 * d_signum(dp1);
}

__device__ void d_compat_orient_intrinsic_2(
    float3 q0, float3 n0, float3 q1, float3 n1,
    float3 &out0, float3 &out1)
{
    float3 q1r = d_rotate_vector_into_plane(q1, n1, n0);
    out0 = q0;
    out1 = q1r * d_signum(dot3(q1r, q0));
}

__device__ void d_compat_orient_intrinsic_6(
    float3 q0, float3 n0, float3 q1, float3 n1,
    float3 &out0, float3 &out1)
{
    float3 q1r = d_rotate_vector_into_plane(q1, n1, n0);
    float3 n0_neg = make_f3(-n0.x, -n0.y, -n0.z);
    float3 t[3] = { d_rotate60(q1r, n0_neg), q1r, d_rotate60(q1r, n0) };
    float dp[3] = { dot3(t[0], q0), dot3(t[1], q0), dot3(t[2], q0) };
    float adp[3] = { fabsf(dp[0]), fabsf(dp[1]), fabsf(dp[2]) };

    out0 = q0;
    if (adp[0] >= adp[1] && adp[0] >= adp[2])
        out1 = t[0] * d_signum(dp[0]);
    else if (adp[1] >= adp[0] && adp[1] >= adp[2])
        out1 = t[1] * d_signum(dp[1]);
    else
        out1 = t[2] * d_signum(dp[2]);
}

// ============================================================
// Device compat functions for position (extrinsic)
// ============================================================

__device__ float3 d_middle_point(float3 p0, float3 n0, float3 p1, float3 n1) {
    float n0p0 = dot3(n0, p0), n0p1 = dot3(n0, p1);
    float n1p0 = dot3(n1, p0), n1p1 = dot3(n1, p1);
    float n0n1 = dot3(n0, n1);
    float denom = 1.0f / (1.0f - n0n1*n0n1 + 1e-4f);
    float lambda_0 = 2.0f*(n0p1 - n0p0 - n0n1*(n1p0 - n1p1))*denom;
    float lambda_1 = 2.0f*(n1p0 - n1p1 - n0n1*(n0p1 - n0p0))*denom;
    return 0.5f*(p0+p1) - 0.25f*(n0*lambda_0 + n1*lambda_1);
}

__device__ float3 d_position_floor_4(float3 o, float3 q, float3 n, float3 p,
                                      float scale, float inv_scale) {
    float3 t = cross3(n, q);
    float3 d = p - o;
    return o + q * floorf(dot3(q, d) * inv_scale) * scale
             + t * floorf(dot3(t, d) * inv_scale) * scale;
}

__device__ float3 d_position_round_4(float3 o, float3 q, float3 n, float3 p,
                                      float scale, float inv_scale) {
    float3 t = cross3(n, q);
    float3 d = p - o;
    return o + q * roundf(dot3(q, d) * inv_scale) * scale
             + t * roundf(dot3(t, d) * inv_scale) * scale;
}

__device__ float3 d_position_round_3(float3 o, float3 q, float3 n, float3 p,
                                      float scale, float inv_scale) {
    float3 t = d_rotate60(q, n);
    float3 d = p - o;
    float dpq = dot3(q, d), dpt = dot3(t, d);
    float u = floorf(( 4*dpq - 2*dpt) * (1.0f / 3.0f) * inv_scale);
    float v = floorf((-2*dpq + 4*dpt) * (1.0f / 3.0f) * inv_scale);

    float best_cost = 1e30f;
    int best_i = 0;
    for (int i = 0; i < 4; ++i) {
        float3 ot = o + (q*(u+(i&1)) + t*(v+((i&2)>>1))) * scale;
        float cost = dot3(ot-p, ot-p);
        if (cost < best_cost) { best_i = i; best_cost = cost; }
    }
    return o + (q*(u+(best_i&1)) + t*(v+((best_i&2)>>1))) * scale;
}

__device__ float3 d_position_floor_3(float3 o, float3 q, float3 n, float3 p,
                                      float scale, float inv_scale) {
    float3 t = d_rotate60(q, n);
    float3 d = p - o;
    float dpq = dot3(q, d), dpt = dot3(t, d);
    float u = floorf(( 4*dpq - 2*dpt) * (1.0f / 3.0f) * inv_scale);
    float v = floorf((-2*dpq + 4*dpt) * (1.0f / 3.0f) * inv_scale);
    return o + (q*u + t*v) * scale;
}

// Extrinsic position compat 4
__device__ void d_compat_pos_extrinsic_4(
    float3 p0, float3 n0, float3 q0, float3 o0,
    float3 p1, float3 n1, float3 q1, float3 o1,
    float scale, float inv_scale,
    float3 &out0, float3 &out1)
{
    float3 t0 = cross3(n0, q0), t1 = cross3(n1, q1);
    float3 mid = d_middle_point(p0, n0, p1, n1);
    float3 o0p = d_position_floor_4(o0, q0, n0, mid, scale, inv_scale);
    float3 o1p = d_position_floor_4(o1, q1, n1, mid, scale, inv_scale);

    float best_cost = 1e30f;
    int best_i = 0, best_j = 0;
    for (int i = 0; i < 4; ++i) {
        float3 o0t = o0p + (q0*float(i&1) + t0*float((i&2)>>1)) * scale;
        for (int j = 0; j < 4; ++j) {
            float3 o1t = o1p + (q1*float(j&1) + t1*float((j&2)>>1)) * scale;
            float3 diff = o0t - o1t;
            float cost = dot3(diff, diff);
            if (cost < best_cost) {
                best_i = i; best_j = j; best_cost = cost;
            }
        }
    }

    out0 = o0p + (q0*float(best_i&1) + t0*float((best_i&2)>>1)) * scale;
    out1 = o1p + (q1*float(best_j&1) + t1*float((best_j&2)>>1)) * scale;
}

// Extrinsic position compat 3
__device__ void d_compat_pos_extrinsic_3(
    float3 p0, float3 n0, float3 q0, float3 o0,
    float3 p1, float3 n1, float3 q1, float3 o1,
    float scale, float inv_scale,
    float3 &out0, float3 &out1)
{
    float3 mid = d_middle_point(p0, n0, p1, n1);
    float3 o0f = d_position_floor_3(o0, q0, n0, mid, scale, inv_scale);
    float3 o1f = d_position_floor_3(o1, q1, n1, mid, scale, inv_scale);
    float3 t0 = d_rotate60(q0, n0), t1 = d_rotate60(q1, n1);

    float best_cost = 1e30f;
    int best_i = 0, best_j = 0;
    for (int i = 0; i < 4; ++i) {
        float3 o0t = o0f + (q0*float(i&1) + t0*float((i&2)>>1)) * scale;
        for (int j = 0; j < 4; ++j) {
            float3 o1t = o1f + (q1*float(j&1) + t1*float((j&2)>>1)) * scale;
            float3 diff = o0t - o1t;
            float cost = dot3(diff, diff);
            if (cost < best_cost) {
                best_i = i; best_j = j; best_cost = cost;
            }
        }
    }
    out0 = o0f + (q0*float(best_i&1) + t0*float((best_i&2)>>1)) * scale;
    out1 = o1f + (q1*float(best_j&1) + t1*float((best_j&2)>>1)) * scale;
}

// Intrinsic position compat 4
__device__ void d_compat_pos_intrinsic_4(
    float3 p0, float3 n0, float3 q0, float3 o0,
    float3 p1, float3 n1, float3 q1, float3 o1,
    float scale, float inv_scale,
    float3 &out0, float3 &out1)
{
    float cosTheta = dot3(n1, n0);
    if (cosTheta < 0.9999f) {
        float3 axis = cross3(n1, n0);
        float factor = (1.0f - cosTheta) / (dot3(axis, axis) + 1e-10f);
        float3 mid = d_middle_point(p0, n0, p1, n1);
        o1 = o1 - mid;
        q1 = q1 * cosTheta + cross3(axis, q1) + axis * (dot3(axis, q1) * factor);
        o1 = o1 * cosTheta + cross3(axis, o1) + axis * (dot3(axis, o1) * factor) + mid;
    }
    out0 = o0;
    out1 = d_position_round_4(o1, q1, n0, o0, scale, inv_scale);
}

// Intrinsic position compat 3
__device__ void d_compat_pos_intrinsic_3(
    float3 p0, float3 n0, float3 q0, float3 o0,
    float3 p1, float3 n1, float3 q1, float3 o1,
    float scale, float inv_scale,
    float3 &out0, float3 &out1)
{
    float cosTheta = dot3(n1, n0);
    if (cosTheta < 0.9999f) {
        float3 axis = cross3(n1, n0);
        float factor = (1.0f - cosTheta) / (dot3(axis, axis) + 1e-10f);
        float3 mid = d_middle_point(p0, n0, p1, n1);
        o1 = o1 - mid;
        q1 = q1 * cosTheta + cross3(axis, q1) + axis * (dot3(axis, q1) * factor);
        o1 = o1 * cosTheta + cross3(axis, o1) + axis * (dot3(axis, o1) * factor) + mid;
    }
    out0 = o0;
    out1 = d_position_round_3(o1, q1, n0, o0, scale, inv_scale);
}

// ============================================================
// Compat dispatch by mode (rosy/posy + extrinsic/intrinsic)
// Encoded as: orient_mode = rosy * 2 + (extrinsic ? 0 : 1)
//             pos_mode = posy * 2 + (extrinsic ? 0 : 1)
// ============================================================

#define ORIENT_EXT2  4
#define ORIENT_INT2  5
#define ORIENT_EXT4  8
#define ORIENT_INT4  9
#define ORIENT_EXT6  12
#define ORIENT_INT6  13

#define POS_EXT3  6
#define POS_INT3  7
#define POS_EXT4  8
#define POS_INT4  9

// ============================================================
// Orientation optimization kernel
// ============================================================

__global__ void kernel_optimize_orientations(
    const uint32_t *phase_indices, uint32_t phase_size,
    const uint32_t *adj_row, const uint32_t *adj_col, const float *adj_weight,
    const float *N, const float *CQ, const float *CQw,
    float *Q,
    int orient_mode)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= phase_size) return;

    uint32_t i = phase_indices[tid];

    float3 n_i = make_f3(N[i*3], N[i*3+1], N[i*3+2]);
    float3 sum = make_f3(Q[i*3], Q[i*3+1], Q[i*3+2]);
    float weight_sum = 0.0f;

    uint32_t row_start = adj_row[i];
    uint32_t row_end = adj_row[i+1];

    for (uint32_t idx = row_start; idx < row_end; ++idx) {
        uint32_t j = adj_col[idx];
        float weight = adj_weight[idx];
        if (weight == 0.0f) continue;

        float3 n_j = make_f3(N[j*3], N[j*3+1], N[j*3+2]);
        float3 q_j = make_f3(Q[j*3], Q[j*3+1], Q[j*3+2]);

        float3 v0, v1;
        switch (orient_mode) {
            case ORIENT_EXT2: d_compat_orient_extrinsic_2(sum, n_i, q_j, n_j, v0, v1); break;
            case ORIENT_INT2: d_compat_orient_intrinsic_2(sum, n_i, q_j, n_j, v0, v1); break;
            case ORIENT_EXT4: d_compat_orient_extrinsic_4(sum, n_i, q_j, n_j, v0, v1); break;
            case ORIENT_INT4: d_compat_orient_intrinsic_4(sum, n_i, q_j, n_j, v0, v1); break;
            case ORIENT_EXT6: d_compat_orient_extrinsic_6(sum, n_i, q_j, n_j, v0, v1); break;
            case ORIENT_INT6: d_compat_orient_intrinsic_6(sum, n_i, q_j, n_j, v0, v1); break;
        }

        sum = v0 * weight_sum + v1 * weight;
        sum = sum - n_i * dot3(n_i, sum);
        weight_sum += weight;

        float n = norm3(sum);
        if (n > 1e-38f) sum = sum * (1.0f / n);
    }

    // Apply constraints
    if (CQw != nullptr && CQw[i] != 0.0f) {
        float cw = CQw[i];
        float3 cq = make_f3(CQ[i*3], CQ[i*3+1], CQ[i*3+2]);

        float3 v0, v1;
        switch (orient_mode) {
            case ORIENT_EXT2: d_compat_orient_extrinsic_2(sum, n_i, cq, n_i, v0, v1); break;
            case ORIENT_INT2: d_compat_orient_intrinsic_2(sum, n_i, cq, n_i, v0, v1); break;
            case ORIENT_EXT4: d_compat_orient_extrinsic_4(sum, n_i, cq, n_i, v0, v1); break;
            case ORIENT_INT4: d_compat_orient_intrinsic_4(sum, n_i, cq, n_i, v0, v1); break;
            case ORIENT_EXT6: d_compat_orient_extrinsic_6(sum, n_i, cq, n_i, v0, v1); break;
            case ORIENT_INT6: d_compat_orient_intrinsic_6(sum, n_i, cq, n_i, v0, v1); break;
        }

        sum = v0 * (1.0f - cw) + v1 * cw;
        sum = sum - n_i * dot3(n_i, sum);
        float n = norm3(sum);
        if (n > 1e-38f) sum = sum * (1.0f / n);
    }

    if (weight_sum > 0.0f) {
        Q[i*3]   = sum.x;
        Q[i*3+1] = sum.y;
        Q[i*3+2] = sum.z;
    }
}

// ============================================================
// Position optimization kernel
// ============================================================

__global__ void kernel_optimize_positions(
    const uint32_t *phase_indices, uint32_t phase_size,
    const uint32_t *adj_row, const uint32_t *adj_col, const float *adj_weight,
    const float *V, const float *N, const float *Q_field,
    const float *CQ, const float *CO, const float *COw,
    float *O,
    float scale, float inv_scale,
    int pos_mode)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= phase_size) return;

    uint32_t i = phase_indices[tid];

    float3 n_i = make_f3(N[i*3], N[i*3+1], N[i*3+2]);
    float3 v_i = make_f3(V[i*3], V[i*3+1], V[i*3+2]);
    float3 q_i = normalize3(make_f3(Q_field[i*3], Q_field[i*3+1], Q_field[i*3+2]));
    float3 sum = make_f3(O[i*3], O[i*3+1], O[i*3+2]);
    float weight_sum = 0.0f;

    uint32_t row_start = adj_row[i];
    uint32_t row_end = adj_row[i+1];

    for (uint32_t idx = row_start; idx < row_end; ++idx) {
        uint32_t j = adj_col[idx];
        float weight = adj_weight[idx];
        if (weight == 0.0f) continue;

        float3 n_j = make_f3(N[j*3], N[j*3+1], N[j*3+2]);
        float3 v_j = make_f3(V[j*3], V[j*3+1], V[j*3+2]);
        float3 q_j = normalize3(make_f3(Q_field[j*3], Q_field[j*3+1], Q_field[j*3+2]));
        float3 o_j = make_f3(O[j*3], O[j*3+1], O[j*3+2]);

        float3 c0, c1;
        switch (pos_mode) {
            case POS_EXT4: d_compat_pos_extrinsic_4(v_i, n_i, q_i, sum, v_j, n_j, q_j, o_j, scale, inv_scale, c0, c1); break;
            case POS_INT4: d_compat_pos_intrinsic_4(v_i, n_i, q_i, sum, v_j, n_j, q_j, o_j, scale, inv_scale, c0, c1); break;
            case POS_EXT3: d_compat_pos_extrinsic_3(v_i, n_i, q_i, sum, v_j, n_j, q_j, o_j, scale, inv_scale, c0, c1); break;
            case POS_INT3: d_compat_pos_intrinsic_3(v_i, n_i, q_i, sum, v_j, n_j, q_j, o_j, scale, inv_scale, c0, c1); break;
        }

        sum = c0 * weight_sum + c1 * weight;
        weight_sum += weight;
        if (weight_sum > 1e-38f)
            sum = sum * (1.0f / weight_sum);
        sum = sum - n_i * dot3(n_i, sum - v_i);
    }

    // Apply constraints
    if (COw != nullptr && COw[i] != 0.0f) {
        float cw = COw[i];
        float3 co = make_f3(CO[i*3], CO[i*3+1], CO[i*3+2]);
        float3 cq = make_f3(CQ[i*3], CQ[i*3+1], CQ[i*3+2]);
        float3 d = co - sum;
        d = d - cq * dot3(cq, d);
        sum = sum + d * cw;
        sum = sum - n_i * dot3(n_i, sum - v_i);
    }

    if (weight_sum > 0.0f) {
        // Round to grid
        float3 result;
        if (pos_mode == POS_EXT4 || pos_mode == POS_INT4)
            result = d_position_round_4(sum, q_i, n_i, v_i, scale, inv_scale);
        else
            result = d_position_round_3(sum, q_i, n_i, v_i, scale, inv_scale);

        O[i*3]   = result.x;
        O[i*3+1] = result.y;
        O[i*3+2] = result.z;
    }
}

// ============================================================
// Propagation kernel (coarse to fine)
// ============================================================

__global__ void kernel_propagate_orient(
    const float *src_Q, const uint32_t *toUpper, uint32_t nCoarse,
    float *dst_Q, const float *dst_N)
{
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= nCoarse) return;

    float3 q = make_f3(src_Q[j*3], src_Q[j*3+1], src_Q[j*3+2]);

    for (int k = 0; k < 2; ++k) {
        uint32_t dest = toUpper[j*2 + k];
        if (dest == 0xFFFFFFFFu) continue;
        float3 n = make_f3(dst_N[dest*3], dst_N[dest*3+1], dst_N[dest*3+2]);
        float3 result = q - n * dot3(n, q);
        dst_Q[dest*3]   = result.x;
        dst_Q[dest*3+1] = result.y;
        dst_Q[dest*3+2] = result.z;
    }
}

__global__ void kernel_propagate_pos(
    const float *src_O, const uint32_t *toUpper, uint32_t nCoarse,
    float *dst_O, const float *dst_N, const float *dst_V)
{
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= nCoarse) return;

    float3 o = make_f3(src_O[j*3], src_O[j*3+1], src_O[j*3+2]);

    for (int k = 0; k < 2; ++k) {
        uint32_t dest = toUpper[j*2 + k];
        if (dest == 0xFFFFFFFFu) continue;
        float3 n = make_f3(dst_N[dest*3], dst_N[dest*3+1], dst_N[dest*3+2]);
        float3 v = make_f3(dst_V[dest*3], dst_V[dest*3+1], dst_V[dest*3+2]);
        float3 result = o - n * dot3(n, o - v);
        dst_O[dest*3]   = result.x;
        dst_O[dest*3+1] = result.y;
        dst_O[dest*3+2] = result.z;
    }
}

// ============================================================
// Host-side data structure for GPU-resident hierarchy
// ============================================================

struct CUDAHierarchyLevel {
    // CSR adjacency
    uint32_t *d_adj_row;     // [nVerts+1]
    uint32_t *d_adj_col;     // [nnz]
    float    *d_adj_weight;  // [nnz]

    // Geometry
    float    *d_V;           // [nVerts*3]
    float    *d_N;           // [nVerts*3]

    // Fields
    float    *d_Q;           // [nVerts*3]
    float    *d_O;           // [nVerts*3]

    // Constraints
    float    *d_CQ;          // [nVerts*3] or null
    float    *d_CO;          // [nVerts*3] or null
    float    *d_CQw;         // [nVerts] or null
    float    *d_COw;         // [nVerts] or null

    // Tree connectivity (to finer level)
    uint32_t *d_toUpper;     // [nVerts*2] or null

    // Phase indices
    std::vector<uint32_t*> d_phases;
    std::vector<uint32_t>  phase_sizes;

    uint32_t nVerts;
    uint32_t nnz;
};

struct CUDAHierarchy {
    std::vector<CUDAHierarchyLevel> levels;
    bool allocated;

    CUDAHierarchy() : allocated(false) {}
    void free();
};

void CUDAHierarchy::free() {
    for (auto& lev : levels) {
        cudaFree(lev.d_adj_row);
        cudaFree(lev.d_adj_col);
        cudaFree(lev.d_adj_weight);
        cudaFree(lev.d_V);
        cudaFree(lev.d_N);
        cudaFree(lev.d_Q);
        cudaFree(lev.d_O);
        if (lev.d_CQ) cudaFree(lev.d_CQ);
        if (lev.d_CO) cudaFree(lev.d_CO);
        if (lev.d_CQw) cudaFree(lev.d_CQw);
        if (lev.d_COw) cudaFree(lev.d_COw);
        if (lev.d_toUpper) cudaFree(lev.d_toUpper);
        for (auto p : lev.d_phases)
            cudaFree(p);
    }
    levels.clear();
    allocated = false;
}

// ============================================================
// Host-side upload / download
// ============================================================

// Forward declare the types we'll use from hierarchy.h via opaque pointers
// (we can't include Eigen from .cu easily, so we use raw float pointers)

extern "C" {

struct CUDAOptimizerContext {
    CUDAHierarchy hier;
    int orient_mode;
    int pos_mode;
    float scale;
    float inv_scale;
};

CUDAOptimizerContext* cuda_optimizer_create() {
    return new CUDAOptimizerContext();
}

void cuda_optimizer_destroy(CUDAOptimizerContext* ctx) {
    if (ctx) {
        ctx->hier.free();
        delete ctx;
    }
}

// Upload hierarchy to GPU
// Arrays are column-major Eigen layout: V[row + col*3] for 3xN matrix
void cuda_optimizer_upload(CUDAOptimizerContext* ctx,
    int nLevels,
    // Per-level data (arrays of pointers)
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
    const uint32_t *phase_counts,  // flattened: [level0_nPhases, level1_nPhases, ...]
    const uint32_t *phase_sizes_flat,  // flattened phase sizes
    int rosy, int posy, bool extrinsic,
    float scale)
{
    ctx->hier.free();
    ctx->orient_mode = rosy * 2 + (extrinsic ? 0 : 1);
    ctx->pos_mode = posy * 2 + (extrinsic ? 0 : 1);
    ctx->scale = scale;
    ctx->inv_scale = 1.0f / scale;
    ctx->hier.levels.resize(nLevels);

    uint32_t phase_offset = 0;

    for (int l = 0; l < nLevels; ++l) {
        CUDAHierarchyLevel& lev = ctx->hier.levels[l];
        uint32_t nV = level_nVerts[l];
        uint32_t nnz = level_nnz[l];
        lev.nVerts = nV;
        lev.nnz = nnz;

        // Adjacency CSR
        cudaMalloc(&lev.d_adj_row, (nV+1) * sizeof(uint32_t));
        cudaMalloc(&lev.d_adj_col, nnz * sizeof(uint32_t));
        cudaMalloc(&lev.d_adj_weight, nnz * sizeof(float));
        cudaMemcpy(lev.d_adj_row, adj_row_ptrs[l], (nV+1)*sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(lev.d_adj_col, adj_col_ptrs[l], nnz*sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(lev.d_adj_weight, adj_weight_ptrs[l], nnz*sizeof(float), cudaMemcpyHostToDevice);

        // Geometry
        cudaMalloc(&lev.d_V, nV*3*sizeof(float));
        cudaMalloc(&lev.d_N, nV*3*sizeof(float));
        cudaMemcpy(lev.d_V, V_ptrs[l], nV*3*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(lev.d_N, N_ptrs[l], nV*3*sizeof(float), cudaMemcpyHostToDevice);

        // Fields
        cudaMalloc(&lev.d_Q, nV*3*sizeof(float));
        cudaMalloc(&lev.d_O, nV*3*sizeof(float));
        cudaMemcpy(lev.d_Q, Q_ptrs[l], nV*3*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(lev.d_O, O_ptrs[l], nV*3*sizeof(float), cudaMemcpyHostToDevice);

        // Constraints
        if (CQw_ptrs[l] && CQ_ptrs[l]) {
            cudaMalloc(&lev.d_CQ, nV*3*sizeof(float));
            cudaMalloc(&lev.d_CQw, nV*sizeof(float));
            cudaMemcpy(lev.d_CQ, CQ_ptrs[l], nV*3*sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(lev.d_CQw, CQw_ptrs[l], nV*sizeof(float), cudaMemcpyHostToDevice);
        } else {
            lev.d_CQ = nullptr;
            lev.d_CQw = nullptr;
        }
        if (COw_ptrs[l] && CO_ptrs[l]) {
            cudaMalloc(&lev.d_CO, nV*3*sizeof(float));
            cudaMalloc(&lev.d_COw, nV*sizeof(float));
            cudaMemcpy(lev.d_CO, CO_ptrs[l], nV*3*sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(lev.d_COw, COw_ptrs[l], nV*sizeof(float), cudaMemcpyHostToDevice);
        } else {
            lev.d_CO = nullptr;
            lev.d_COw = nullptr;
        }

        // Tree: toUpper(l) is 2 x nVerts(l+1), maps coarse level l+1 -> fine level l
        if (toUpper_ptrs[l] && (l + 1) < nLevels) {
            uint32_t nCoarse = level_nVerts[l + 1];
            cudaMalloc(&lev.d_toUpper, nCoarse*2*sizeof(uint32_t));
            cudaMemcpy(lev.d_toUpper, toUpper_ptrs[l], nCoarse*2*sizeof(uint32_t), cudaMemcpyHostToDevice);
        } else {
            lev.d_toUpper = nullptr;
        }

        // Phases
        uint32_t nPhases = phase_counts[l];
        lev.d_phases.resize(nPhases);
        lev.phase_sizes.resize(nPhases);
        for (uint32_t p = 0; p < nPhases; ++p) {
            uint32_t ps = phase_sizes_flat[phase_offset];
            lev.phase_sizes[p] = ps;
            cudaMalloc(&lev.d_phases[p], ps * sizeof(uint32_t));
            cudaMemcpy(lev.d_phases[p], phase_index_ptrs[phase_offset],
                       ps * sizeof(uint32_t), cudaMemcpyHostToDevice);
            phase_offset++;
        }
    }

    ctx->hier.allocated = true;
}

// Run orientation optimization on GPU for one level
void cuda_optimize_orientations(CUDAOptimizerContext* ctx, int level) {
    CUDAHierarchyLevel& lev = ctx->hier.levels[level];
    const int BS = 256;

    for (uint32_t p = 0; p < lev.d_phases.size(); ++p) {
        uint32_t ps = lev.phase_sizes[p];
        if (ps == 0) continue;
        int grid = (ps + BS - 1) / BS;
        kernel_optimize_orientations<<<grid, BS>>>(
            lev.d_phases[p], ps,
            lev.d_adj_row, lev.d_adj_col, lev.d_adj_weight,
            lev.d_N, lev.d_CQ, lev.d_CQw,
            lev.d_Q,
            ctx->orient_mode);
        cudaGetLastError();
    }
    cudaDeviceSynchronize();
}

// Run position optimization on GPU for one level
void cuda_optimize_positions(CUDAOptimizerContext* ctx, int level) {
    CUDAHierarchyLevel& lev = ctx->hier.levels[level];
    const int BS = 256;

    for (uint32_t p = 0; p < lev.d_phases.size(); ++p) {
        uint32_t ps = lev.phase_sizes[p];
        if (ps == 0) continue;
        int grid = (ps + BS - 1) / BS;
        kernel_optimize_positions<<<grid, BS>>>(
            lev.d_phases[p], ps,
            lev.d_adj_row, lev.d_adj_col, lev.d_adj_weight,
            lev.d_V, lev.d_N, lev.d_Q,
            lev.d_CQ, lev.d_CO, lev.d_COw,
            lev.d_O,
            ctx->scale, ctx->inv_scale,
            ctx->pos_mode);
    }
    cudaDeviceSynchronize();
}

// Propagate orientation field from coarse to fine
// fine.d_toUpper stores mRes.toUpper(fine_level) which maps level fine_level+1 -> fine_level
void cuda_propagate_orient(CUDAOptimizerContext* ctx, int fine_level) {
    int coarse_level = fine_level + 1;
    CUDAHierarchyLevel& coarse = ctx->hier.levels[coarse_level];
    CUDAHierarchyLevel& fine = ctx->hier.levels[fine_level];

    if (!fine.d_toUpper) return;

    const int BS = 256;
    int grid = (coarse.nVerts + BS - 1) / BS;
    kernel_propagate_orient<<<grid, BS>>>(
        coarse.d_Q, fine.d_toUpper, coarse.nVerts,
        fine.d_Q, fine.d_N);
    cudaDeviceSynchronize();
}

// Propagate position field from coarse to fine
void cuda_propagate_pos(CUDAOptimizerContext* ctx, int fine_level) {
    int coarse_level = fine_level + 1;
    CUDAHierarchyLevel& coarse = ctx->hier.levels[coarse_level];
    CUDAHierarchyLevel& fine = ctx->hier.levels[fine_level];

    if (!fine.d_toUpper) return;

    const int BS = 256;
    int grid = (coarse.nVerts + BS - 1) / BS;
    kernel_propagate_pos<<<grid, BS>>>(
        coarse.d_O, fine.d_toUpper, coarse.nVerts,
        fine.d_O, fine.d_N, fine.d_V);
    cudaDeviceSynchronize();
}

// Download Q field from GPU back to host
void cuda_download_Q(CUDAOptimizerContext* ctx, int level, float *Q_host) {
    CUDAHierarchyLevel& lev = ctx->hier.levels[level];
    cudaMemcpy(Q_host, lev.d_Q, lev.nVerts*3*sizeof(float), cudaMemcpyDeviceToHost);
}

// Download O field from GPU back to host
void cuda_download_O(CUDAOptimizerContext* ctx, int level, float *O_host) {
    CUDAHierarchyLevel& lev = ctx->hier.levels[level];
    cudaMemcpy(O_host, lev.d_O, lev.nVerts*3*sizeof(float), cudaMemcpyDeviceToHost);
}

// Download all Q and O fields from GPU
void cuda_download_all_fields(CUDAOptimizerContext* ctx,
    float **Q_ptrs, float **O_ptrs, int nLevels)
{
    for (int l = 0; l < nLevels; ++l) {
        CUDAHierarchyLevel& lev = ctx->hier.levels[l];
        cudaMemcpy(Q_ptrs[l], lev.d_Q, lev.nVerts*3*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(O_ptrs[l], lev.d_O, lev.nVerts*3*sizeof(float), cudaMemcpyDeviceToHost);
    }
}

// Run full hierarchical orientation optimization (coarse to fine, 6 iterations per level)
// Matches CPU Optimizer::run(): propagate only one level down after each level's iterations
void cuda_optimize_orientations_full(CUDAOptimizerContext* ctx, int nLevels) {
    const int levelIterations = 6;

    for (int level = nLevels - 1; level >= 0; --level) {
        for (int iter = 0; iter < levelIterations; ++iter) {
            cuda_optimize_orientations(ctx, level);
        }
        // Propagate one level down (from current level to level-1)
        if (level > 0) {
            cuda_propagate_orient(ctx, level - 1);
        }
    }
}

// Run full hierarchical position optimization (coarse to fine, 6 iterations per level)
void cuda_optimize_positions_full(CUDAOptimizerContext* ctx, int nLevels) {
    const int levelIterations = 6;

    for (int level = nLevels - 1; level >= 0; --level) {
        for (int iter = 0; iter < levelIterations; ++iter) {
            cuda_optimize_positions(ctx, level);
        }
        if (level > 0) {
            cuda_propagate_pos(ctx, level - 1);
        }
    }
}

} // extern "C"
