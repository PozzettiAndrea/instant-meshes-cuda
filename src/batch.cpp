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
#include "bvh.h"
#include "checkpoint.h"

#ifdef WITH_CUDA
#include "optimizer_cuda.h"
#include "init_kernels.h"
#endif

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
static void cuda_run_optimization(MultiResolutionHierarchy &mRes,
                                  int rosy, int posy, bool extrinsic,
                                  bool do_orient, bool do_position) {
    int nLevels = mRes.levels();
    if (nLevels == 0) return;

    // Convert adjacency to flat CSR per level
    std::vector<FlatCSR> csrs(nLevels);
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

        csrs[l] = adjacency_to_csr(mRes.adj(l), nV);
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

    // Run optimization
    if (do_orient) {
        cuda_optimize_orientations_full(ctx, nLevels);
    }

    if (do_position) {
        // Re-upload Q fields since orientation may have changed them
        // (they're already on GPU if we just ran orient)
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

            // Subdivision still runs on CPU (priority-queue based, inherently sequential)
            if (stats.mMaximumEdgeLength*2 > scale || stats.mMaximumEdgeLength > stats.mAverageEdgeLength * 2) {
                cout << "Input mesh is too coarse for the desired output edge length "
                        "(max input mesh edge length=" << stats.mMaximumEdgeLength
                     << "), subdividing .." << endl;
                build_dedge(F, V, V2E, E2E, boundary, nonManifold);
                subdivide(F, V, V2E, E2E, boundary, nonManifold,
                          std::min(scale/2, (Float) stats.mAverageEdgeLength*2), opts.deterministic);
            }

#ifdef WITH_CUDA
            if (opts.optim_strategy == 1 && opts.creaseAngle < 0) {
                // GPU path: build dedge + normals + areas in one GPU round-trip
                uint32_t nV = V.cols(), nF = F.cols();
                V2E.resize(nV);
                E2E.resize(3 * nF);
                // Temporary uint32_t arrays for boundary/nonManifold (GPU uses uint32_t, CPU uses bool)
                std::vector<uint32_t> bnd_u32(nV), nm_u32(nV);
                N.resize(3, nV);
                A.resize(nV);

                cuda_init_mesh(
                    F.data(), nF, V.data(), nV,
                    V2E.data(), E2E.data(),
                    bnd_u32.data(), nm_u32.data(),
                    N.data(), A.data());

                // Convert uint32_t flags to VectorXb
                boundary.resize(nV);
                nonManifold.resize(nV);
                for (uint32_t i = 0; i < nV; ++i) {
                    boundary[i] = bnd_u32[i] != 0;
                    nonManifold[i] = nm_u32[i] != 0;
                }

                // Still need CPU adjacency matrix (pointer-based Link** format)
                adj = generate_adjacency_matrix_uniform(F, V2E, E2E, nonManifold);
            } else
#endif
            {
                // CPU path
                build_dedge(F, V, V2E, E2E, boundary, nonManifold);
                adj = generate_adjacency_matrix_uniform(F, V2E, E2E, nonManifold);

                if (opts.creaseAngle >= 0)
                    generate_crease_normals(F, V, V2E, E2E, boundary, nonManifold,
                                            opts.creaseAngle, N, crease_in);
                else
                    generate_smooth_normals(F, V, V2E, E2E, nonManifold, N);

                compute_dual_vertex_areas(F, V, V2E, E2E, nonManifold, A);
            }

            mRes.setE2E(std::move(E2E));
        }

        cout << "[TIMING] post-subdivide: " << timeString(pipeline_timer.value() - t_stage) << endl;
        if (should_stop(STAGE_POST_SUBDIVIDE)) goto done;
    }

    // ============================================================
    // STAGE: post-hierarchy
    // ============================================================
    t_stage = pipeline_timer.value();

    if (should_run(STAGE_POST_HIERARCHY)) {
        mRes.setAdj(std::move(adj));
        mRes.setF(std::move(F));
        mRes.setV(std::move(V));
        mRes.setA(std::move(A));
        mRes.setN(std::move(N));
        mRes.setScale(scale);
        mRes.build(opts.deterministic);
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
    // STAGE: post-orient (orientation field optimization)
    // ============================================================
    t_stage = pipeline_timer.value();

    if (should_run(STAGE_POST_ORIENT)) {
        cout << "Optimizing orientation field .. ";
        cout.flush();

#ifdef WITH_CUDA
        if (opts.optim_strategy == 1) {
            cuda_run_optimization(mRes, opts.rosy, opts.posy, opts.extrinsic,
                                  true /* orient */, false /* position */);
            cout << "done (CUDA). (took " << timeString(pipeline_timer.value() - t_stage) << ")" << endl;
        } else
#endif
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
        maybe_save(mRes, STAGE_POST_ORIENT, hdr);
        if (should_stop(STAGE_POST_ORIENT)) goto done;
    }

    // ============================================================
    // STAGE: post-position (position field optimization)
    // ============================================================
    t_stage = pipeline_timer.value();

    if (should_run(STAGE_POST_POSITION)) {
        cout << "Optimizing position field .. ";
        cout.flush();

#ifdef WITH_CUDA
        if (opts.optim_strategy == 1) {
            cuda_run_optimization(mRes, opts.rosy, opts.posy, opts.extrinsic,
                                  false /* orient */, true /* position */);
            cout << "done (CUDA). (took " << timeString(pipeline_timer.value() - t_stage) << ")" << endl;
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
            cout << "done (CPU). (took " << timeString(pipeline_timer.value() - t_stage) << ")" << endl;
        }

        cout << "[TIMING] post-position: " << timeString(pipeline_timer.value() - t_stage) << endl;
        maybe_save(mRes, STAGE_POST_POSITION, hdr);
        if (should_stop(STAGE_POST_POSITION)) goto done;
    }

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
