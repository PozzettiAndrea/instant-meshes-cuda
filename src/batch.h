/*
    batch.h -- command line interface to Instant Meshes (CUDA-accelerated)

    This file is part of the implementation of

        Instant Field-Aligned Meshes
        Wenzel Jakob, Daniele Panozzo, Marco Tarini, and Olga Sorkine-Hornung
        In ACM Transactions on Graphics (Proc. SIGGRAPH Asia 2015)

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#pragma once

#include "common.h"
#include "checkpoint.h"
#include <string>

struct BatchOptions {
    std::string input;
    std::string output;
    int rosy = 4;
    int posy = 4;
    Float scale = -1;
    int face_count = -1;
    int vertex_count = -1;
    Float creaseAngle = -1;
    bool extrinsic = true;
    bool align_to_boundaries = false;
    int smooth_iter = 2;
    int knn_points = 10;
    bool pure_quad = true;
    bool deterministic = false;
    // Strategy flags (like QuadriFlow's -ff, -subdiv, -dse)
    int optim_strategy = 0;     // -optim: 0=cpu, 1=cuda
    int hierarchy_strategy = 0; // -hier: 0=cpu (future: 1=cuda)
    int extract_strategy = 0;   // -extract: 0=cpu (future: 1=cuda)

    // Checkpoint options
    std::string save_dir;
    PipelineStage save_at = STAGE_NONE;
    bool save_all = false;
    PipelineStage run_from = STAGE_NONE;
    PipelineStage run_to = STAGE_NONE;
};

extern void batch_process(const BatchOptions &opts);
