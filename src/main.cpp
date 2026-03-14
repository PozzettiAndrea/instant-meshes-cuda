/*
    main.cpp -- Instant Meshes application entry point (CUDA-accelerated)

    This file is part of the implementation of

        Instant Field-Aligned Meshes
        Wenzel Jakob, Daniele Panozzo, Marco Tarini, and Olga Sorkine-Hornung
        In ACM Transactions on Graphics (Proc. SIGGRAPH Asia 2015)

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "batch.h"
#include "viewer.h"
#include "serializer.h"
#include <thread>
#include <cstdlib>

/* Force usage of discrete GPU on laptops */
NANOGUI_FORCE_DISCRETE_GPU();

int nprocs = -1;

int main(int argc, char **argv) {
    std::vector<std::string> args;
    bool fullscreen = false, help = false, compat = false;
    BatchOptions opts;

    #if defined(__APPLE__)
        bool launched_from_finder = false;
    #endif

    try {
        for (int i=1; i<argc; ++i) {
            if (strcmp("--fullscreen", argv[i]) == 0 || strcmp("-F", argv[i]) == 0) {
                fullscreen = true;
            } else if (strcmp("--help", argv[i]) == 0 || strcmp("-h", argv[i]) == 0) {
                help = true;
            } else if (strcmp("--deterministic", argv[i]) == 0 || strcmp("-d", argv[i]) == 0) {
                opts.deterministic = true;
            } else if (strcmp("--intrinsic", argv[i]) == 0 || strcmp("-i", argv[i]) == 0) {
                opts.extrinsic = false;
            } else if (strcmp("--boundaries", argv[i]) == 0 || strcmp("-b", argv[i]) == 0) {
                opts.align_to_boundaries = true;
            } else if (strcmp("--threads", argv[i]) == 0 || strcmp("-t", argv[i]) == 0) {
                if (++i >= argc) { cerr << "Missing thread count!" << endl; return -1; }
                nprocs = str_to_uint32_t(argv[i]);
            } else if (strcmp("--smooth", argv[i]) == 0 || strcmp("-S", argv[i]) == 0) {
                if (++i >= argc) { cerr << "Missing smoothing iteration count!" << endl; return -1; }
                opts.smooth_iter = str_to_uint32_t(argv[i]);
            } else if (strcmp("--knn", argv[i]) == 0 || strcmp("-k", argv[i]) == 0) {
                if (++i >= argc) { cerr << "Missing knn point count!" << endl; return -1; }
                opts.knn_points = str_to_uint32_t(argv[i]);
            } else if (strcmp("--crease", argv[i]) == 0 || strcmp("-c", argv[i]) == 0) {
                if (++i >= argc) { cerr << "Missing crease angle!" << endl; return -1; }
                opts.creaseAngle = str_to_float(argv[i]);
            } else if (strcmp("--rosy", argv[i]) == 0 || strcmp("-r", argv[i]) == 0) {
                if (++i >= argc) { cerr << "Missing rotation symmetry type!" << endl; return -1; }
                opts.rosy = str_to_int32_t(argv[i]);
            } else if (strcmp("--posy", argv[i]) == 0 || strcmp("-p", argv[i]) == 0) {
                if (++i >= argc) { cerr << "Missing position symmetry type!" << endl; return -1; }
                opts.posy = str_to_int32_t(argv[i]);
                if (opts.posy == 6) opts.posy = 3;
            } else if (strcmp("--scale", argv[i]) == 0 || strcmp("-s", argv[i]) == 0) {
                if (++i >= argc) { cerr << "Missing scale!" << endl; return -1; }
                opts.scale = str_to_float(argv[i]);
            } else if (strcmp("--faces", argv[i]) == 0 || strcmp("-f", argv[i]) == 0) {
                if (++i >= argc) { cerr << "Missing face count!" << endl; return -1; }
                opts.face_count = str_to_int32_t(argv[i]);
            } else if (strcmp("--vertices", argv[i]) == 0 || strcmp("-v", argv[i]) == 0) {
                if (++i >= argc) { cerr << "Missing vertex count!" << endl; return -1; }
                opts.vertex_count = str_to_int32_t(argv[i]);
            } else if (strcmp("--output", argv[i]) == 0 || strcmp("-o", argv[i]) == 0) {
                if (++i >= argc) { cerr << "Missing output file!" << endl; return -1; }
                opts.output = argv[i];
            } else if (strcmp("--dominant", argv[i]) == 0 || strcmp("-D", argv[i]) == 0) {
                opts.pure_quad = false;
            } else if (strcmp("--compat", argv[i]) == 0 || strcmp("-C", argv[i]) == 0) {
                compat = true;
            // ---- Strategy flags (like QuadriFlow -ff/-subdiv/-dse) ----
            } else if (strcmp("-optim", argv[i]) == 0 || strcmp("--optim", argv[i]) == 0) {
                if (++i >= argc) { cerr << "Missing optim strategy!" << endl; return -1; }
                if (strcmp(argv[i], "cpu") == 0) opts.optim_strategy = 0;
                else if (strcmp(argv[i], "cuda") == 0) opts.optim_strategy = 1;
                else { cerr << "Unknown -optim strategy: " << argv[i] << " (valid: cpu, cuda)" << endl; return -1; }
            } else if (strcmp("-hier", argv[i]) == 0 || strcmp("--hier", argv[i]) == 0) {
                if (++i >= argc) { cerr << "Missing hier strategy!" << endl; return -1; }
                if (strcmp(argv[i], "cpu") == 0) opts.hierarchy_strategy = 0;
                else if (strcmp(argv[i], "cuda") == 0) opts.hierarchy_strategy = 1;
                else { cerr << "Unknown -hier strategy: " << argv[i] << " (valid: cpu, cuda)" << endl; return -1; }
            } else if (strcmp("-extract", argv[i]) == 0 || strcmp("--extract", argv[i]) == 0) {
                if (++i >= argc) { cerr << "Missing extract strategy!" << endl; return -1; }
                if (strcmp(argv[i], "cpu") == 0) opts.extract_strategy = 0;
                else if (strcmp(argv[i], "cuda") == 0) opts.extract_strategy = 1;
                else { cerr << "Unknown -extract strategy: " << argv[i] << " (valid: cpu, cuda)" << endl; return -1; }
            } else if (strcmp("--cuda", argv[i]) == 0 || strcmp("-G", argv[i]) == 0) {
                // Shorthand: enable CUDA for all strategies that support it
                opts.optim_strategy = 1;
            // ---- Checkpoint flags ----
            } else if (strcmp("-save-dir", argv[i]) == 0 || strcmp("--save-dir", argv[i]) == 0) {
                if (++i >= argc) { cerr << "Missing save directory!" << endl; return -1; }
                opts.save_dir = argv[i];
            } else if (strcmp("-save-at", argv[i]) == 0 || strcmp("--save-at", argv[i]) == 0) {
                if (++i >= argc) { cerr << "Missing stage name!" << endl; return -1; }
                opts.save_at = stage_from_name(argv[i]);
                if (opts.save_at == STAGE_NONE) {
                    cerr << "Unknown stage: " << argv[i] << endl;
                    list_stages();
                    return -1;
                }
            } else if (strcmp("-save-all", argv[i]) == 0 || strcmp("--save-all", argv[i]) == 0) {
                opts.save_all = true;
            } else if (strcmp("-run-from", argv[i]) == 0 || strcmp("--run-from", argv[i]) == 0) {
                if (++i >= argc) { cerr << "Missing stage name!" << endl; return -1; }
                opts.run_from = stage_from_name(argv[i]);
                if (opts.run_from == STAGE_NONE) {
                    cerr << "Unknown stage: " << argv[i] << endl;
                    list_stages();
                    return -1;
                }
            } else if (strcmp("-run-to", argv[i]) == 0 || strcmp("--run-to", argv[i]) == 0) {
                if (++i >= argc) { cerr << "Missing stage name!" << endl; return -1; }
                opts.run_to = stage_from_name(argv[i]);
                if (opts.run_to == STAGE_NONE) {
                    cerr << "Unknown stage: " << argv[i] << endl;
                    list_stages();
                    return -1;
                }
            } else if (strcmp("-list-stages", argv[i]) == 0 || strcmp("--list-stages", argv[i]) == 0) {
                list_stages();
                return 0;
#if defined(__APPLE__)
            } else if (strncmp("-psn", argv[i], 4) == 0) {
                launched_from_finder = true;
#endif
            } else {
                if (strncmp(argv[i], "-", 1) == 0) {
                    cerr << "Invalid argument: \"" << argv[i] << "\"!" << endl;
                    help = true;
                }
                args.push_back(argv[i]);
            }
        }
    } catch (const std::exception &e) {
        cout << "Error: " << e.what() << endl;
        help = true;
    }

    if ((opts.posy != 3 && opts.posy != 4) || (opts.rosy != 2 && opts.rosy != 4 && opts.rosy != 6)) {
        cerr << "Error: Invalid symmetry type!" << endl;
        help = true;
    }

    int nConstraints = 0;
    nConstraints += opts.scale > 0 ? 1 : 0;
    nConstraints += opts.face_count > 0 ? 1 : 0;
    nConstraints += opts.vertex_count > 0 ? 1 : 0;

    if (nConstraints > 1) {
        cerr << "Error: Only one of --scale, --faces and --vertices can be used at once!" << endl;
        help = true;
    }

    if (args.size() > 1 || help || (!opts.output.empty() && args.size() == 0 && opts.run_from == STAGE_NONE)) {
        cout << "Syntax: " << argv[0] << " [options] <input mesh / point cloud / application state snapshot>" << endl;
        cout << "Options:" << endl;
        cout << "   -o, --output <output>       Writes to the specified PLY/OBJ output file in batch mode" << endl;
        cout << "   -t, --threads <count>       Number of threads used for parallel computations" << endl;
        cout << "   -d, --deterministic         Prefer (slower) deterministic algorithms" << endl;
        cout << "   -c, --crease <degrees>      Dihedral angle threshold for creases" << endl;
        cout << "   -S, --smooth <iter>         Number of smoothing & ray tracing reprojection steps (default: 2)" << endl;
        cout << "   -D, --dominant              Generate a tri/quad dominant mesh instead of a pure tri/quad mesh" << endl;
        cout << "   -i, --intrinsic             Intrinsic mode (extrinsic is the default)" << endl;
        cout << "   -b, --boundaries            Align to boundaries (only applies when the mesh is not closed)" << endl;
        cout << "   -r, --rosy <number>         Specifies the orientation symmetry type (2, 4, or 6)" << endl;
        cout << "   -p, --posy <number>         Specifies the position symmetry type (4 or 6)" << endl;
        cout << "   -s, --scale <scale>         Desired world space length of edges in the output" << endl;
        cout << "   -f, --faces <count>         Desired face count of the output mesh" << endl;
        cout << "   -v, --vertices <count>      Desired vertex count of the output mesh" << endl;
        cout << "   -C, --compat                Compatibility mode to load snapshots from old software versions" << endl;
        cout << "   -k, --knn <count>           Point cloud mode: number of adjacent points to consider" << endl;
        cout << "   -F, --fullscreen            Open a full-screen window" << endl;
#ifdef WITH_CUDA
        cout << "   -G, --cuda                  Shorthand: enable CUDA for all supported strategies" << endl;
#endif
        cout << endl;
        cout << "Strategy options (like QuadriFlow -ff/-subdiv/-dse):" << endl;
        cout << "   --optim <cpu|cuda>          Field optimization strategy (default: cpu)" << endl;
        cout << "   --hier <cpu|cuda>           Hierarchy build strategy (default: cpu, cuda=future)" << endl;
        cout << "   --extract <cpu|cuda>        Mesh extraction strategy (default: cpu, cuda=future)" << endl;
        cout << endl;
        cout << "Checkpoint options:" << endl;
        cout << "   --save-dir <dir>            Directory for checkpoint files" << endl;
        cout << "   --save-at <stage>           Save checkpoint after this specific stage" << endl;
        cout << "   --save-all                  Save checkpoint after every stage" << endl;
        cout << "   --run-from <stage>          Load checkpoint and resume from this stage" << endl;
        cout << "   --run-to <stage>            Stop execution after this stage" << endl;
        cout << "   --list-stages               Print all pipeline stage names" << endl;
        cout << endl;
        cout << "   -h, --help                  Display this message" << endl;
        return -1;
    }

    if (args.size() == 0 && opts.run_from == STAGE_NONE)
        cout << "Running in GUI mode, start with -h for instructions on batch mode." << endl;

    if (args.size() > 0)
        opts.input = args[0];

    tbb::task_scheduler_init init(nprocs == -1 ? tbb::task_scheduler_init::automatic : nprocs);

    if (!opts.output.empty() || opts.run_from != STAGE_NONE || !opts.save_dir.empty()) {
        try {
            batch_process(opts);
            return 0;
        } catch (const std::exception &e) {
            cerr << "Caught runtime error : " << e.what() << endl;
            return -1;
        }
    }

    try {
        nanogui::init();

        #if defined(__APPLE__)
            if (launched_from_finder)
                nanogui::chdir_to_bundle_parent();
        #endif

        {
            nanogui::ref<Viewer> viewer = new Viewer(fullscreen, opts.deterministic);
            viewer->setVisible(true);

            if (args.size() == 1) {
                if (Serializer::isSerializedFile(args[0])) {
                    viewer->loadState(args[0], compat);
                } else {
                    viewer->loadInput(args[0], opts.creaseAngle,
                            opts.scale, opts.face_count, opts.vertex_count,
                            opts.rosy, opts.posy, opts.knn_points);
                    viewer->setExtrinsic(opts.extrinsic);
                }
            }

            nanogui::mainloop();
        }

        nanogui::shutdown();
    } catch (const std::runtime_error &e) {
        std::string error_msg = std::string("Caught a fatal error: ") + std::string(e.what());
        #if defined(_WIN32)
            MessageBoxA(nullptr, error_msg.c_str(), NULL, MB_ICONERROR | MB_OK);
        #else
            std::cerr << error_msg << endl;
        #endif
        return -1;
    }

    return EXIT_SUCCESS;
}
