/*
    checkpoint.h -- Pipeline checkpoint system for Instant Meshes CUDA

    Allows saving/loading full pipeline state at any stage boundary,
    enabling fast benchmarking of individual stages without re-running
    the full pipeline.

    Usage:
      Full run with saves:  ./InstantMeshes input.obj -o out.obj -f 10000 \
                               -save-all -save-dir /tmp/checkpoints
      Resume from stage:    ./InstantMeshes input.obj -o out.obj \
                               -run-from post-orient -save-dir /tmp/checkpoints
      Run single stage:     ./InstantMeshes input.obj -o out.obj \
                               -run-from post-orient -run-to post-position \
                               -save-dir /tmp/checkpoints

    Stage names (in pipeline order):
      post-load        After mesh loading + stats + scale computation
      post-subdivide   After subdivision + dedge + adjacency + normals + areas
      post-hierarchy   After hierarchy build + solution init + constraints + BVH
      post-orient      After orientation field optimization
      post-position    After position field optimization
      post-extract     After graph extraction + face extraction
*/

#pragma once

#include "common.h"
#include <string>

struct MultiResolutionHierarchy;

// Pipeline stages (in order)
enum PipelineStage {
    STAGE_NONE = -1,
    STAGE_POST_LOAD = 0,
    STAGE_POST_SUBDIVIDE,
    STAGE_POST_HIERARCHY,
    STAGE_POST_ORIENT,
    STAGE_POST_POSITION,
    STAGE_POST_EXTRACT,
    STAGE_COUNT
};

// Convert stage name string to enum
PipelineStage stage_from_name(const char* name);

// Convert enum to display name
const char* stage_name(PipelineStage s);

// Print all stage names
void list_stages();

// Checkpoint file header
struct CheckpointHeader {
    char magic[4];          // "IMC\0" (Instant Meshes Checkpoint)
    int version;            // format version (1)
    char stage[64];         // stage name
    int rosy;               // rotation symmetry type
    int posy;               // position symmetry type
    int face_count;         // target face count
    int vertex_count;       // target vertex count
    int extrinsic;          // extrinsic mode
    int align_to_boundaries;
    int deterministic;
    int pure_quad;
    int smooth_iter;
    float scale;            // target edge length
    float crease_angle;
    int optim_strategy;     // -optim flag: 0=cpu, 1=cuda
    int hierarchy_strategy; // -hier flag: 0=cpu, 1=cuda
    int extract_strategy;   // -extract flag: 0=cpu, 1=cuda
    char input_mesh[256];   // input mesh path
    long long timestamp;    // unix timestamp
    char reserved[116];     // padding for future use
};

// Save full pipeline state at a given stage
void save_checkpoint(const MultiResolutionHierarchy &mRes, PipelineStage stage,
                     const char* dir, const CheckpointHeader &params);

// Load checkpoint, returns stage that was saved
PipelineStage load_checkpoint(MultiResolutionHierarchy &mRes, const char* dir,
                              PipelineStage stage);

// Check if a checkpoint file exists for a given stage
bool checkpoint_exists(const char* dir, PipelineStage stage);
