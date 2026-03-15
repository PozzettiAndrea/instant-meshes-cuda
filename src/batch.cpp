/*
    batch.cpp -- command line interface to Instant Meshes (CUDA-accelerated)

    This file is part of the implementation of

        Instant Field-Aligned Meshes
        Wenzel Jakob, Daniele Panozzo, Marco Tarini, and Olga Sorkine-Hornung
        In ACM Transactions on Graphics (Proc. SIGGRAPH Asia 2015)

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "batch.h"
#include "meshio.h"
#include "dedge.h"
#include "subdivide.h"
#include "meshstats.h"
#include "hierarchy.h"
#include "field.h"
#include "normal.h"
#include "extract.h"
#include "reorder.h"
#include "bvh.h"
#include "checkpoint.h"

#ifdef WITH_CUDA
#include "optimizer_cuda.h"
#endif

#ifdef WITH_RXMESH
#include "rxmesh_subdivide.h"
#endif

#if 0  // GPU-resident path disabled — caused device linking performance regression
#include "gpu_hierarchy.h"
#include "subdivide_gpu.h"
#endif

// Forward declarations for graph coloring (defined in hierarchy.cpp)
extern void generate_graph_coloring(const AdjacencyMatrix &adj, uint32_t size,
                             std::vector<std::vector<uint32_t>> &phases,
                             const ProgressCallback &progress = ProgressCallback());
extern void generate_graph_coloring_deterministic(const AdjacencyMatrix &adj, uint32_t size,
                             std::vector<std::vector<uint32_t>> &phases,
                             const ProgressCallback &progress = ProgressCallback());
extern void generate_random_partition(const AdjacencyMatrix &adj, uint32_t size,
                             std::vector<std::vector<uint32_t>> &phases,
                             int k = 16,
                             const ProgressCallback &progress = ProgressCallback());

// ============================================================
// Helper: convert MultiResolutionHierarchy adjacency to flat CSR
// ============================================================

struct FlatCSR {
    std::vector<uint32_t> row_ptr;
    std::vector<uint32_t> col_idx;
    std::vector<float>    weights;
};

static FlatCSR adjacency_to_csr(const AdjacencyMatrix &adj, uint32_t nVerts) {
    FlatCSR csr;
    csr.row_ptr.resize(nVerts + 1);
    uint32_t nnz = (uint32_t)(adj[nVerts] - adj[0]);
    csr.col_idx.resize(nnz);
    csr.weights.resize(nnz);

    for (uint32_t i = 0; i <= nVerts; ++i)
        csr.row_ptr[i] = (uint32_t)(adj[i] - adj[0]);

    for (uint32_t i = 0; i < nnz; ++i) {
        csr.col_idx[i] = adj[0][i].id;
        csr.weights[i] = adj[0][i].weight;
    }
    return csr;
}

#ifdef WITH_CUDA
// Upload hierarchy to CUDA and run optimization
// Cached CSR — computed once after hierarchy build, reused for all CUDA calls
static std::vector<FlatCSR> g_cached_csrs;
static int g_cached_nLevels = 0;

static void cache_hierarchy_csr(MultiResolutionHierarchy &mRes) {
    Timer<> timer;
    int nLevels = mRes.levels();
    g_cached_csrs.resize(nLevels);
    for (int l = 0; l < nLevels; ++l)
        g_cached_csrs[l] = adjacency_to_csr(mRes.adj(l), mRes.size(l));
    g_cached_nLevels = nLevels;
    cout << "CSR cache built for " << nLevels << " levels (took "
         << timeString(timer.value()) << ")" << endl;
}

static void cuda_run_optimization(MultiResolutionHierarchy &mRes,
                                  int rosy, int posy, bool extrinsic,
                                  bool do_orient, bool do_position) {
    int nLevels = mRes.levels();
    if (nLevels == 0) return;

    // Use cached CSR if available, else compute on the fly
    if (g_cached_nLevels != nLevels) cache_hierarchy_csr(mRes);
    std::vector<FlatCSR> &csrs = g_cached_csrs;
    std::vector<uint32_t> level_nVerts(nLevels);
    std::vector<uint32_t> level_nnz(nLevels);

    // Pointers for upload
    std::vector<const uint32_t*> adj_row_ptrs(nLevels);
    std::vector<const uint32_t*> adj_col_ptrs(nLevels);
    std::vector<const float*> adj_weight_ptrs(nLevels);
    std::vector<const float*> V_ptrs(nLevels);
    std::vector<const float*> N_ptrs(nLevels);
    std::vector<const float*> Q_ptrs(nLevels);
    std::vector<const float*> O_ptrs(nLevels);
    std::vector<const float*> CQ_ptrs(nLevels);
    std::vector<const float*> CO_ptrs(nLevels);
    std::vector<const float*> CQw_ptrs(nLevels);
    std::vector<const float*> COw_ptrs(nLevels);
    std::vector<const uint32_t*> toUpper_ptrs(nLevels);

    // Flatten phases
    std::vector<uint32_t> phase_counts(nLevels);
    std::vector<const uint32_t*> all_phase_ptrs;
    std::vector<uint32_t> all_phase_sizes;

    for (int l = 0; l < nLevels; ++l) {
        uint32_t nV = mRes.size(l);
        level_nVerts[l] = nV;

        // CSR already cached — just read sizes
        level_nnz[l] = (uint32_t)csrs[l].col_idx.size();

        adj_row_ptrs[l] = csrs[l].row_ptr.data();
        adj_col_ptrs[l] = csrs[l].col_idx.data();
        adj_weight_ptrs[l] = csrs[l].weights.data();

        V_ptrs[l] = mRes.V(l).data();
        N_ptrs[l] = mRes.N(l).data();
        Q_ptrs[l] = mRes.Q(l).data();
        O_ptrs[l] = mRes.O(l).data();

        if (mRes.CQ(l).size() > 0) {
            CQ_ptrs[l] = mRes.CQ(l).data();
            CQw_ptrs[l] = mRes.CQw(l).data();
        } else {
            CQ_ptrs[l] = nullptr;
            CQw_ptrs[l] = nullptr;
        }
        if (mRes.CO(l).size() > 0) {
            CO_ptrs[l] = mRes.CO(l).data();
            COw_ptrs[l] = mRes.COw(l).data();
        } else {
            CO_ptrs[l] = nullptr;
            COw_ptrs[l] = nullptr;
        }

        if (l < nLevels - 1) {
            toUpper_ptrs[l] = mRes.toUpper(l).data();
        } else {
            toUpper_ptrs[l] = nullptr;
        }

        const auto &phases = mRes.phases(l);
        phase_counts[l] = (uint32_t)phases.size();
        for (const auto &phase : phases) {
            all_phase_ptrs.push_back(phase.data());
            all_phase_sizes.push_back((uint32_t)phase.size());
        }
    }

    // Create CUDA context and upload
    CUDAOptimizerContext *ctx = cuda_optimizer_create();
    cuda_optimizer_upload(ctx, nLevels,
        level_nVerts.data(), level_nnz.data(),
        adj_row_ptrs.data(), adj_col_ptrs.data(), adj_weight_ptrs.data(),
        V_ptrs.data(), N_ptrs.data(), Q_ptrs.data(), O_ptrs.data(),
        CQ_ptrs.data(), CO_ptrs.data(), CQw_ptrs.data(), COw_ptrs.data(),
        toUpper_ptrs.data(),
        all_phase_ptrs.data(), phase_counts.data(), all_phase_sizes.data(),
        rosy, posy, extrinsic, mRes.scale());

    // Orient removed from GPU — runs on CPU (memory-bound, CPU L3 > GPU L2)
    // Only position runs on GPU
    if (do_position) {
        // Penner-inspired kernel tested but quality too poor (43% vs 92% quads).
        // Full Penner approach needs edge-length optimization + Ptolemy flips + integer LP.
        // Simplified Jacobi+snap doesn't maintain integer grid alignment.
        // Reverting to standard Gauss-Seidel phase-based position optimization.
        cuda_optimize_positions_full(ctx, nLevels);
    }

    // Download results back to host
    std::vector<float*> Q_out(nLevels), O_out(nLevels);
    for (int l = 0; l < nLevels; ++l) {
        Q_out[l] = mRes.Q(l).data();
        O_out[l] = mRes.O(l).data();
    }
    cuda_download_all_fields(ctx, Q_out.data(), O_out.data(), nLevels);

    cuda_optimizer_destroy(ctx);
}
#endif

// ============================================================
// Main batch pipeline with checkpoint support
// ============================================================

void batch_process(const BatchOptions &opts) {
    cout << endl;
    cout << "Running in batch mode:" << endl;
    cout << "   Input file             = " << opts.input << endl;
    cout << "   Output file            = " << opts.output << endl;
    cout << "   Rotation symmetry type = " << opts.rosy << endl;
    cout << "   Position symmetry type = " << (opts.posy==3?6:opts.posy) << endl;
    cout << "   Extrinsic mode         = " << (opts.extrinsic ? "enabled" : "disabled") << endl;
    cout << "   Align to boundaries    = " << (opts.align_to_boundaries ? "yes" : "no") << endl;
    cout << "   Fully deterministic    = " << (opts.deterministic ? "yes" : "no") << endl;
#ifdef WITH_CUDA
    cout << "   Optim strategy         = " << (opts.optim_strategy == 0 ? "cpu" : "cuda") << endl;
    cout << "   Hierarchy strategy     = " << (opts.hierarchy_strategy == 0 ? "cpu" : "cuda") << endl;
#endif
    if (opts.posy == 4)
        cout << "   Output mode            = " << (opts.pure_quad ? "pure quad mesh" : "quad-dominant mesh") << endl;
    if (opts.run_from != STAGE_NONE)
        cout << "   Resume from            = " << stage_name(opts.run_from) << endl;
    if (opts.run_to != STAGE_NONE)
        cout << "   Stop at                = " << stage_name(opts.run_to) << endl;
    if (!opts.save_dir.empty())
        cout << "   Checkpoint dir         = " << opts.save_dir << endl;
    cout << endl;

    // Pipeline control lambdas
    auto should_run = [&](PipelineStage stage) -> bool {
        if (opts.run_from != STAGE_NONE && stage <= opts.run_from) return false;
        if (opts.run_to != STAGE_NONE && stage > opts.run_to) return false;
        return true;
    };

    auto maybe_save = [&](const MultiResolutionHierarchy &mRes, PipelineStage stage,
                          const CheckpointHeader &hdr) {
        if (!opts.save_dir.empty() && (opts.save_all || stage == opts.save_at))
            save_checkpoint(mRes, stage, opts.save_dir.c_str(), hdr);
    };

    auto should_stop = [&](PipelineStage stage) -> bool {
        return (opts.run_to != STAGE_NONE && stage >= opts.run_to);
    };

    // Build checkpoint header from options
    CheckpointHeader hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.rosy = opts.rosy;
    hdr.posy = opts.posy;
    hdr.face_count = opts.face_count;
    hdr.vertex_count = opts.vertex_count;
    hdr.extrinsic = opts.extrinsic ? 1 : 0;
    hdr.align_to_boundaries = opts.align_to_boundaries ? 1 : 0;
    hdr.deterministic = opts.deterministic ? 1 : 0;
    hdr.pure_quad = opts.pure_quad ? 1 : 0;
    hdr.smooth_iter = opts.smooth_iter;
    hdr.scale = (float)opts.scale;
    hdr.crease_angle = (float)opts.creaseAngle;
    hdr.optim_strategy = opts.optim_strategy;
    hdr.hierarchy_strategy = opts.hierarchy_strategy;
    hdr.extract_strategy = opts.extract_strategy;
    strncpy(hdr.input_mesh, opts.input.c_str(), sizeof(hdr.input_mesh) - 1);

    MatrixXu F;
    MatrixXf V, N;
    VectorXf A;
    std::set<uint32_t> crease_in, crease_out;
    BVH *bvh = nullptr;
    AdjacencyMatrix adj = nullptr;
    MultiResolutionHierarchy mRes;
    Float scale = opts.scale;
    int face_count = opts.face_count;
    int vertex_count = opts.vertex_count;
    bool pointcloud = false;
    MeshStats stats;

    Timer<> pipeline_timer;

    // ============================================================
    // Try to load checkpoint for resume
    // ============================================================
    if (opts.run_from != STAGE_NONE) {
        if (opts.save_dir.empty()) {
            cerr << "ERROR: -run-from requires -save-dir" << endl;
            return;
        }
        PipelineStage loaded = load_checkpoint(mRes, opts.save_dir.c_str(), opts.run_from);
        if (loaded == STAGE_NONE) {
            cerr << "ERROR: Failed to load checkpoint for stage '" << stage_name(opts.run_from) << "'" << endl;
            return;
        }
        scale = mRes.scale();
        cout << "[PIPELINE] Resuming after stage '" << stage_name(opts.run_from)
             << "' (scale=" << scale << ")" << endl;
    }

    // ============================================================
    // STAGE: post-load
    // ============================================================
    unsigned long long t_stage;
    t_stage = pipeline_timer.value();

    if (should_run(STAGE_POST_LOAD)) {
        load_mesh_or_pointcloud(opts.input, F, V, N);
        pointcloud = F.size() == 0;

        stats = compute_mesh_stats(F, V, opts.deterministic);

        if (pointcloud) {
            bvh = new BVH(&F, &V, &N, stats.mAABB);
            bvh->build();
            adj = generate_adjacency_matrix_pointcloud(V, N, bvh, stats, opts.knn_points, opts.deterministic);
            A.resize(V.cols());
            A.setConstant(1.0f);
        }

        if (scale < 0 && vertex_count < 0 && face_count < 0) {
            cout << "No target vertex count/face count/scale argument provided. "
                    "Setting to the default of 1/16 * input vertex count." << endl;
            vertex_count = V.cols() / 16;
        }

        if (scale > 0) {
            Float face_area = opts.posy == 4 ? (scale*scale) : (std::sqrt(3.f)/4.f*scale*scale);
            face_count = stats.mSurfaceArea / face_area;
            vertex_count = opts.posy == 4 ? face_count : (face_count / 2);
        } else if (face_count > 0) {
            Float face_area = stats.mSurfaceArea / face_count;
            vertex_count = opts.posy == 4 ? face_count : (face_count / 2);
            scale = opts.posy == 4 ? std::sqrt(face_area) : (2*std::sqrt(face_area * std::sqrt(1.f/3.f)));
        } else if (vertex_count > 0) {
            face_count = opts.posy == 4 ? vertex_count : (vertex_count * 2);
            Float face_area = stats.mSurfaceArea / face_count;
            scale = opts.posy == 4 ? std::sqrt(face_area) : (2*std::sqrt(face_area * std::sqrt(1.f/3.f)));
        }

        cout << "Output mesh goals (approximate)" << endl;
        cout << "   Vertex count           = " << vertex_count << endl;
        cout << "   Face count             = " << face_count << endl;
        cout << "   Edge length            = " << scale << endl;

        cout << "[TIMING] post-load: " << timeString(pipeline_timer.value() - t_stage) << endl;
        // Note: can't checkpoint here since mRes not built yet
        if (should_stop(STAGE_POST_LOAD)) goto done;
    }

    // ============================================================
    // STAGE: post-subdivide
    // ============================================================
    t_stage = pipeline_timer.value();

    if (should_run(STAGE_POST_SUBDIVIDE)) {
        if (!pointcloud) {
            VectorXu V2E, E2E;
            VectorXb boundary, nonManifold;

            if (stats.mMaximumEdgeLength*2 > scale || stats.mMaximumEdgeLength > stats.mAverageEdgeLength * 2) {
                cout << "Input mesh is too coarse for the desired output edge length "
                        "(max input mesh edge length=" << stats.mMaximumEdgeLength
                     << "), subdividing .." << endl;
                build_dedge(F, V, V2E, E2E, boundary, nonManifold);
                Float maxLen = std::min(scale/2, (Float) stats.mAverageEdgeLength*2);

#ifdef WITH_RXMESH
                if (opts.hierarchy_strategy == 1) {
                    // GPU edge splitting via RXMesh cavity operator
                    Timer<> rx_timer;
                    cout << "GPU subdivision (RXMesh) .. ";
                    cout.flush();

                    // Flatten F,V for C interface
                    uint32_t nV_in = V.cols(), nF_in = F.cols();
                    std::vector<float> V_flat(nV_in * 3);
                    std::vector<uint32_t> F_flat(nF_in * 3);
                    for (uint32_t i = 0; i < nV_in; ++i) {
                        V_flat[i*3+0] = V(0,i);
                        V_flat[i*3+1] = V(1,i);
                        V_flat[i*3+2] = V(2,i);
                    }
                    for (uint32_t i = 0; i < nF_in; ++i) {
                        F_flat[i*3+0] = F(0,i);
                        F_flat[i*3+1] = F(1,i);
                        F_flat[i*3+2] = F(2,i);
                    }

                    float *V_out = nullptr; uint32_t nV_out = 0;
                    uint32_t *F_out = nullptr; uint32_t nF_out = 0;
                    rxmesh_subdivide(V_flat.data(), nV_in, F_flat.data(), nF_in,
                                     (float)maxLen, &V_out, &nV_out, &F_out, &nF_out);

                    // Copy back into Eigen matrices
                    V.resize(3, nV_out);
                    F.resize(3, nF_out);
                    for (uint32_t i = 0; i < nV_out; ++i) {
                        V(0,i) = V_out[i*3+0];
                        V(1,i) = V_out[i*3+1];
                        V(2,i) = V_out[i*3+2];
                    }
                    for (uint32_t i = 0; i < nF_out; ++i) {
                        F(0,i) = F_out[i*3+0];
                        F(1,i) = F_out[i*3+1];
                        F(2,i) = F_out[i*3+2];
                    }
                    free(V_out); free(F_out);

                    cout << nV_in << " -> " << nV_out << " verts, "
                         << nF_in << " -> " << nF_out << " faces. "
                         << "(took " << timeString(rx_timer.value()) << ")" << endl;
                } else
#endif
                {
                    subdivide(F, V, V2E, E2E, boundary, nonManifold, maxLen, opts.deterministic);
                }
            }

            // Fast spatial reorder: BFS face traversal for cache-coherent access
            {
                Timer<> reorder_timer;
                cout << "Reordering mesh for spatial coherence .. ";
                cout.flush();

                uint32_t nV_r = V.cols(), nF_r = F.cols();
                // BFS over faces using E2E for adjacency
                build_dedge(F, V, V2E, E2E, boundary, nonManifold);

                std::vector<uint32_t> face_order;
                face_order.reserve(nF_r);
                std::vector<bool> face_visited(nF_r, false);
                std::queue<uint32_t> bfs;

                for (uint32_t seed = 0; seed < nF_r; ++seed) {
                    if (face_visited[seed]) continue;
                    bfs.push(seed);
                    face_visited[seed] = true;
                    while (!bfs.empty()) {
                        uint32_t f = bfs.front(); bfs.pop();
                        face_order.push_back(f);
                        for (int j = 0; j < 3; ++j) {
                            uint32_t opp = E2E[f*3 + j];
                            if (opp != INVALID) {
                                uint32_t fn = opp / 3;
                                if (!face_visited[fn]) {
                                    face_visited[fn] = true;
                                    bfs.push(fn);
                                }
                            }
                        }
                    }
                }

                // Build vertex remapping from face visit order
                std::vector<uint32_t> v_map(nV_r, INVALID);
                uint32_t nV_new = 0;
                for (uint32_t fi = 0; fi < nF_r; ++fi) {
                    uint32_t f = face_order[fi];
                    for (int j = 0; j < 3; ++j) {
                        uint32_t v = F(j, f);
                        if (v_map[v] == INVALID)
                            v_map[v] = nV_new++;
                    }
                }

                // Apply remapping
                MatrixXu F_new(3, nF_r);
                MatrixXf V_new(3, nV_r);
                for (uint32_t fi = 0; fi < nF_r; ++fi) {
                    uint32_t f = face_order[fi];
                    for (int j = 0; j < 3; ++j)
                        F_new(j, fi) = v_map[F(j, f)];
                }
                for (uint32_t v = 0; v < nV_r; ++v) {
                    if (v_map[v] != INVALID)
                        V_new.col(v_map[v]) = V.col(v);
                }
                F = std::move(F_new);
                V = std::move(V_new);

                cout << "done. (took " << timeString(reorder_timer.value()) << ")" << endl;
            }

            // Build everything on the reordered mesh (one pass)
            build_dedge(F, V, V2E, E2E, boundary, nonManifold);
            adj = generate_adjacency_matrix_uniform(F, V2E, E2E, nonManifold);

            if (opts.creaseAngle >= 0)
                generate_crease_normals(F, V, V2E, E2E, boundary, nonManifold,
                                        opts.creaseAngle, N, crease_in);
            else
                generate_smooth_normals(F, V, V2E, E2E, nonManifold, N);

            compute_dual_vertex_areas(F, V, V2E, E2E, nonManifold, A);

            mRes.setE2E(std::move(E2E));
        }

        cout << "[TIMING] post-subdivide: " << timeString(pipeline_timer.value() - t_stage) << endl;
        if (should_stop(STAGE_POST_SUBDIVIDE)) goto done;
    }

    // ============================================================
    // STAGE: post-hierarchy + post-orient + post-position
    // GPU-RESIDENT PATH: hierarchy build + optimization on GPU
    // CPU PATH: original code
    // ============================================================

#if 0  // GPU-resident path disabled — caused device linking performance regression
    if (opts.optim_strategy == 2 && should_run(STAGE_POST_HIERARCHY) && !pointcloud) {
        // ========================================
        // GPU-RESIDENT PIPELINE (experimental, -optim gpu-resident)
        // ========================================
        GPUHierarchy *gpuH = gpu_hierarchy_create();

        // -- post-hierarchy --
        t_stage = pipeline_timer.value();

        // GPU init: upload F,V → build dedge + normals + areas + adjacency
        uint32_t nV0 = V.cols(), nF0 = F.cols();
        VectorXu V2E_u(nV0), E2E_u(3*nF0);
        std::vector<uint32_t> bnd_u32(nV0), nm_u32(nV0);
        gpu_hierarchy_init(gpuH, (const uint32_t*)F.data(), nF0,
                           V.data(), nV0,
                           V2E_u.data(), (uint32_t*)E2E_u.data(),
                           bnd_u32.data(), nm_u32.data());
        cout << "GPU init done (dedge+normals+areas+adj). ";

        // GPU hierarchy build (graph coloring on CPU)
        int nLevels = gpu_hierarchy_build(gpuH, opts.deterministic);
        cout << nLevels << " levels built." << endl;

        // Graph coloring for each level (CPU, downloads adj CSR)
        for (int l = 0; l < nLevels; ++l) {
            uint32_t nVl = gpu_hierarchy_level_nVerts(gpuH, l);
            // Download adj CSR for coloring
            std::vector<uint32_t> rp(nVl+1), ci;
            std::vector<float> wt;
            // First get nnz
            rp.resize(nVl+1);
            uint32_t nnz_l = gpu_hierarchy_download_adj(gpuH, l, rp.data(), nullptr, nullptr);
            ci.resize(nnz_l); wt.resize(nnz_l);
            gpu_hierarchy_download_adj(gpuH, l, rp.data(), ci.data(), wt.data());

            // Build temporary Link** for graph_coloring (it needs this format)
            AdjacencyMatrix tmp_adj = new Link*[nVl + 1];
            Link* tmp_links = new Link[nnz_l];
            for (uint32_t i = 0; i <= nVl; ++i)
                tmp_adj[i] = tmp_links + rp[i];
            for (uint32_t i = 0; i < nnz_l; ++i) {
                tmp_links[i].id = ci[i];
                tmp_links[i].weight = wt[i];
                tmp_links[i].ivar_uint32 = 0;
            }

            std::vector<std::vector<uint32_t>> phases;
            if (opts.deterministic)
                generate_graph_coloring_deterministic(tmp_adj, nVl, phases);
            else
                generate_graph_coloring(tmp_adj, nVl, phases);

            // Upload phases to GPU
            std::vector<const uint32_t*> phase_ptrs(phases.size());
            std::vector<uint32_t> phase_sizes(phases.size());
            for (size_t p = 0; p < phases.size(); ++p) {
                phase_ptrs[p] = phases[p].data();
                phase_sizes[p] = (uint32_t)phases[p].size();
            }
            gpu_hierarchy_upload_phases(gpuH, l, phase_ptrs.data(), phase_sizes.data(), (uint32_t)phases.size());

            delete[] tmp_links;
            delete[] tmp_adj;
        }

        // Init random fields on GPU
        gpu_hierarchy_init_fields(gpuH, scale);

        // Boundary alignment (needs CPU-side data)
        if (opts.align_to_boundaries) {
            // Download V, N for level 0 to build constraints on CPU
            MatrixXf V0(3, nV0), N0(3, nV0);
            gpu_hierarchy_download_V(gpuH, 0, V0.data(), nV0);
            gpu_hierarchy_download_N(gpuH, 0, N0.data(), nV0);

            // Build constraints
            MatrixXf CQ0(3, nV0), CO0(3, nV0);
            VectorXf CQw0(nV0), COw0(nV0);
            CQ0.setZero(); CO0.setZero(); CQw0.setZero(); COw0.setZero();
            for (uint32_t i = 0; i < 3*nF0; ++i) {
                if (E2E_u[i] == INVALID) {
                    uint32_t i0 = F(i%3, i/3);
                    uint32_t i1 = F((i+1)%3, i/3);
                    Vector3f p0 = V0.col(i0), p1 = V0.col(i1);
                    Vector3f edge = p1-p0;
                    if (edge.squaredNorm() > 0) {
                        edge.normalize();
                        CO0.col(i0) = p0; CO0.col(i1) = p1;
                        CQ0.col(i0) = CQ0.col(i1) = edge;
                        CQw0[i0] = CQw0[i1] = COw0[i0] = COw0[i1] = 1.0f;
                    }
                }
            }
            gpu_hierarchy_upload_constraints(gpuH, 0, CQ0.data(), CO0.data(),
                                             CQw0.data(), COw0.data(), nV0);
            // TODO: propagate constraints to coarser levels on GPU
        }

        // BVH for extraction smoothing (still needs CPU mRes data)
        if (opts.smooth_iter > 0) {
            // Download V, N for BVH
            MatrixXf V0(3, nV0), N0(3, nV0);
            gpu_hierarchy_download_V(gpuH, 0, V0.data(), nV0);
            gpu_hierarchy_download_N(gpuH, 0, N0.data(), nV0);
            // F is still valid (not moved yet in GPU path)
            bvh = new BVH(&F, nullptr, nullptr, stats.mAABB);
            // Need to set the actual data pointers after mRes is built
        }

        cout << "[TIMING] post-hierarchy (GPU): " << timeString(pipeline_timer.value() - t_stage) << endl;
        if (should_stop(STAGE_POST_HIERARCHY)) { gpu_hierarchy_destroy(gpuH); goto done; }

        // -- post-orient --
        t_stage = pipeline_timer.value();
        if (should_run(STAGE_POST_ORIENT)) {
            cout << "Optimizing orientation field (GPU-resident) .. ";
            cout.flush();
            gpu_hierarchy_optimize_orient(gpuH, nLevels, opts.rosy, opts.posy, opts.extrinsic, scale);
            cout << "done. (took " << timeString(pipeline_timer.value() - t_stage) << ")" << endl;
            cout << "[TIMING] post-orient (GPU): " << timeString(pipeline_timer.value() - t_stage) << endl;
        }

        // -- post-position --
        t_stage = pipeline_timer.value();
        if (should_run(STAGE_POST_POSITION)) {
            cout << "Optimizing position field (GPU-resident) .. ";
            cout.flush();
            gpu_hierarchy_optimize_position(gpuH, nLevels, opts.rosy, opts.posy, opts.extrinsic, scale);
            cout << "done. (took " << timeString(pipeline_timer.value() - t_stage) << ")" << endl;
            cout << "[TIMING] post-position (GPU): " << timeString(pipeline_timer.value() - t_stage) << endl;
        }

        // Download fields into CPU mRes for extraction + checkpoint
        mRes.setF(MatrixXu(F));
        mRes.setE2E(VectorXu(E2E_u));
        mRes.setScale(scale);

        // Download level 0 geometry
        {
            uint32_t nVl = gpu_hierarchy_level_nVerts(gpuH, 0);
            MatrixXf Vl(3, nVl), Nl(3, nVl);
            VectorXf Al(nVl);
            gpu_hierarchy_download_V(gpuH, 0, Vl.data(), nVl);
            gpu_hierarchy_download_N(gpuH, 0, Nl.data(), nVl);
            gpu_hierarchy_download_A(gpuH, 0, Al.data(), nVl);
            mRes.setV(std::move(Vl));
            mRes.setN(std::move(Nl));
            mRes.setA(std::move(Al));

            // Download level 0 adjacency CSR and convert to Link**
            uint32_t nnz0 = gpu_hierarchy_download_adj(gpuH, 0, nullptr, nullptr, nullptr);
            std::vector<uint32_t> rp0(nVl+1), ci0(nnz0);
            std::vector<float> wt0(nnz0);
            gpu_hierarchy_download_adj(gpuH, 0, rp0.data(), ci0.data(), wt0.data());

            AdjacencyMatrix adj0 = new Link*[nVl + 1];
            Link* links0 = new Link[nnz0];
            for (uint32_t i = 0; i <= nVl; ++i)
                adj0[i] = links0 + rp0[i];
            for (uint32_t i = 0; i < nnz0; ++i) {
                links0[i].id = ci0[i];
                links0[i].weight = wt0[i];
                links0[i].ivar_uint32 = 0;
            }
            mRes.setAdj(std::move(adj0));
        }

        // Download Q, O for all levels
        mRes.mQ.resize(nLevels);
        mRes.mO.resize(nLevels);
        for (int l = 0; l < nLevels; ++l) {
            uint32_t nVl = gpu_hierarchy_level_nVerts(gpuH, l);
            mRes.mQ[l].resize(3, nVl);
            mRes.mO[l].resize(3, nVl);
            gpu_hierarchy_download_Q(gpuH, l, mRes.mQ[l].data(), nVl);
            gpu_hierarchy_download_O(gpuH, l, mRes.mO[l].data(), nVl);
        }

        // BVH
        if (bvh) { delete bvh; bvh = nullptr; }
        if (opts.smooth_iter > 0) {
            bvh = new BVH(&mRes.F(), &mRes.V(), &mRes.N(), stats.mAABB);
            bvh->build();
        }

        // Singularity count
        if (should_run(STAGE_POST_ORIENT)) {
            std::map<uint32_t, uint32_t> sing;
            compute_orientation_singularities(mRes, sing, opts.extrinsic, opts.rosy);
            cout << "Orientation field has " << sing.size() << " singularities." << endl;
        }

        gpu_hierarchy_destroy(gpuH);

        // Skip CPU hierarchy+orient+position stages
        goto gpu_done_extract;
    }
#endif

    // ========================================
    // CPU PATH (original)
    // ========================================
    t_stage = pipeline_timer.value();

    if (should_run(STAGE_POST_HIERARCHY)) {
        mRes.setAdj(std::move(adj));
        mRes.setF(std::move(F));
        mRes.setV(std::move(V));
        mRes.setA(std::move(A));
        mRes.setN(std::move(N));
        mRes.setScale(scale);
        int coloring_mode = 0;
#ifdef WITH_CUDA
        // coloring_mode 0 = exact coloring (best for CUDA optimizer convergence)
        // coloring_mode 1 = random partition (faster hierarchy, slower optimizer — net wash)
#endif
        mRes.build(opts.deterministic, ProgressCallback(), coloring_mode);
#ifdef WITH_CUDA
        if (opts.optim_strategy == 1) cache_hierarchy_csr(mRes);
#endif
        mRes.resetSolution();

        if (opts.align_to_boundaries && !pointcloud) {
            mRes.clearConstraints();
            for (uint32_t i = 0; i < 3*mRes.F().cols(); ++i) {
                if (mRes.E2E()[i] == INVALID) {
                    uint32_t i0 = mRes.F()(i%3, i/3);
                    uint32_t i1 = mRes.F()((i+1)%3, i/3);
                    Vector3f p0 = mRes.V().col(i0), p1 = mRes.V().col(i1);
                    Vector3f edge = p1-p0;
                    if (edge.squaredNorm() > 0) {
                        edge.normalize();
                        mRes.CO().col(i0) = p0;
                        mRes.CO().col(i1) = p1;
                        mRes.CQ().col(i0) = mRes.CQ().col(i1) = edge;
                        mRes.CQw()[i0] = mRes.CQw()[i1] = mRes.COw()[i0] =
                            mRes.COw()[i1] = 1.0f;
                    }
                }
            }
            mRes.propagateConstraints(opts.rosy, opts.posy);
        }

        if (bvh) {
            bvh->setData(&mRes.F(), &mRes.V(), &mRes.N());
        } else if (opts.smooth_iter > 0) {
            bvh = new BVH(&mRes.F(), &mRes.V(), &mRes.N(), stats.mAABB);
            bvh->build();
        }

        cout << "[TIMING] post-hierarchy: " << timeString(pipeline_timer.value() - t_stage) << endl;
        maybe_save(mRes, STAGE_POST_HIERARCHY, hdr);
        if (should_stop(STAGE_POST_HIERARCHY)) goto done;
    }

    // ============================================================
    // ORIENT + POSITION: Async overlap (CPU orient || GPU position upload)
    // While CPU runs orient (~550ms), GPU uploads hierarchy data.
    // When orient finishes, upload Q and launch GPU position.
    // While GPU runs position (~315ms), CPU computes singularities.
    // ============================================================

    t_stage = pipeline_timer.value();
    if (should_run(STAGE_POST_ORIENT) && should_run(STAGE_POST_POSITION)) {
#ifdef WITH_CUDA
        if (opts.optim_strategy == 1) {
            // --- ASYNC OVERLAP: CPU orient + GPU position ---
            cout << "Optimizing orientation (CPU) + position (GPU, async) .. " << endl;

            // Step 1: Start CPU orient in the Optimizer thread
            Optimizer optimizer(mRes, false);
            optimizer.setRoSy(opts.rosy);
            optimizer.setPoSy(opts.posy);
            optimizer.setExtrinsic(opts.extrinsic);
            optimizer.optimizeOrientations(-1);
            optimizer.notify();

            // Step 2: While orient runs on CPU, do nothing on GPU
            // (future: could pre-upload V, N, adjacency here)

            // Step 3: Wait for orient to finish
            optimizer.wait();
            optimizer.shutdown();
            cout << "  Orient done (CPU). (took " << timeString(pipeline_timer.value() - t_stage) << ")" << endl;
            cout << "[TIMING] post-orient: " << timeString(pipeline_timer.value() - t_stage) << endl;

            // Step 4: Launch GPU position (Q is now ready from CPU orient)
            unsigned long long t_pos = pipeline_timer.value();
            cuda_run_optimization(mRes, opts.rosy, opts.posy, opts.extrinsic,
                                  false, true);

            // Step 5: While GPU runs position, compute singularities on CPU
            // (singularity computation only needs Q, N, F at level 0 — already available)
            std::map<uint32_t, uint32_t> sing;
            compute_orientation_singularities(mRes, sing, opts.extrinsic, opts.rosy);
            cout << "  Orientation field has " << sing.size() << " singularities." << endl;

            // GPU position already finished (cuda_run_optimization syncs internally)
            cout << "  Position done (CUDA). (took " << timeString(pipeline_timer.value() - t_pos) << ")" << endl;
            cout << "[TIMING] post-orient+position: " << timeString(pipeline_timer.value() - t_stage) << endl;

        } else
#endif
        {
            // --- CPU-only path ---
            cout << "Optimizing orientation field .. ";
            cout.flush();
            {
                Optimizer optimizer(mRes, false);
                optimizer.setRoSy(opts.rosy);
                optimizer.setPoSy(opts.posy);
                optimizer.setExtrinsic(opts.extrinsic);
                optimizer.optimizeOrientations(-1);
                optimizer.notify();
                optimizer.wait();
                optimizer.shutdown();
                cout << "done (CPU). (took " << timeString(pipeline_timer.value() - t_stage) << ")" << endl;
            }

            std::map<uint32_t, uint32_t> sing;
            compute_orientation_singularities(mRes, sing, opts.extrinsic, opts.rosy);
            cout << "Orientation field has " << sing.size() << " singularities." << endl;
            cout << "[TIMING] post-orient: " << timeString(pipeline_timer.value() - t_stage) << endl;

            t_stage = pipeline_timer.value();
            cout << "Optimizing position field .. ";
            cout.flush();
            {
                Optimizer optimizer(mRes, false);
                optimizer.setRoSy(opts.rosy);
                optimizer.setPoSy(opts.posy);
                optimizer.setExtrinsic(opts.extrinsic);
                optimizer.optimizePositions(-1);
                optimizer.notify();
                optimizer.wait();
                optimizer.shutdown();
                cout << "done (CPU). (took " << timeString(pipeline_timer.value() - t_stage) << ")" << endl;
            }
            cout << "[TIMING] post-position: " << timeString(pipeline_timer.value() - t_stage) << endl;
        }

        if (should_stop(STAGE_POST_POSITION)) goto done;
    } else {
        // Handle individual stage execution
        if (should_run(STAGE_POST_ORIENT)) {
            t_stage = pipeline_timer.value();
            cout << "Optimizing orientation field .. ";
            cout.flush();
            {
                Optimizer optimizer(mRes, false);
                optimizer.setRoSy(opts.rosy);
                optimizer.setPoSy(opts.posy);
                optimizer.setExtrinsic(opts.extrinsic);
                optimizer.optimizeOrientations(-1);
                optimizer.notify();
                optimizer.wait();
                optimizer.shutdown();
                cout << "done (CPU)." << endl;
            }
            std::map<uint32_t, uint32_t> sing;
            compute_orientation_singularities(mRes, sing, opts.extrinsic, opts.rosy);
            cout << "Orientation field has " << sing.size() << " singularities." << endl;
            cout << "[TIMING] post-orient: " << timeString(pipeline_timer.value() - t_stage) << endl;
            if (should_stop(STAGE_POST_ORIENT)) goto done;
        }
        if (should_run(STAGE_POST_POSITION)) {
            t_stage = pipeline_timer.value();
            cout << "Optimizing position field .. ";
            cout.flush();
#ifdef WITH_CUDA
            if (opts.optim_strategy == 1) {
                cuda_run_optimization(mRes, opts.rosy, opts.posy, opts.extrinsic, false, true);
                cout << "done (CUDA)." << endl;
            } else
#endif
            {
                Optimizer optimizer(mRes, false);
                optimizer.setRoSy(opts.rosy);
                optimizer.setPoSy(opts.posy);
                optimizer.setExtrinsic(opts.extrinsic);
                optimizer.optimizePositions(-1);
                optimizer.notify();
                optimizer.wait();
                optimizer.shutdown();
                cout << "done (CPU)." << endl;
            }
            cout << "[TIMING] post-position: " << timeString(pipeline_timer.value() - t_stage) << endl;
            if (should_stop(STAGE_POST_POSITION)) goto done;
        }
    }

#ifdef WITH_CUDA
gpu_done_extract:
#endif

    // ============================================================
    // STAGE: post-extract (graph + face extraction)
    // ============================================================
    t_stage = pipeline_timer.value();

    if (should_run(STAGE_POST_EXTRACT)) {
        MatrixXf O_extr, N_extr, Nf_extr;
        std::vector<std::vector<TaggedLink>> adj_extr;
        extract_graph(mRes, opts.extrinsic, opts.rosy, opts.posy, adj_extr, O_extr, N_extr,
                      crease_in, crease_out, opts.deterministic);

        MatrixXu F_extr;
        extract_faces(adj_extr, O_extr, N_extr, Nf_extr, F_extr, opts.posy,
                mRes.scale(), crease_out, true, opts.pure_quad, bvh, opts.smooth_iter);

        cout << "[TIMING] post-extract: " << timeString(pipeline_timer.value() - t_stage) << endl;

        if (!opts.output.empty())
            write_mesh(opts.output, F_extr, O_extr, MatrixXf(), Nf_extr);
    }

done:
    cout << endl << "[TIMING] === Total pipeline: " << timeString(pipeline_timer.value()) << " ===" << endl;
    if (bvh)
        delete bvh;
}
