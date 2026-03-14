/*
    checkpoint.cpp -- Binary checkpoint save/load for Instant Meshes CUDA

    Serializes the full MultiResolutionHierarchy state to binary files
    at each pipeline stage, enabling stage-by-stage benchmarking and
    fast resume.
*/

#include "checkpoint.h"
#include "hierarchy.h"

#include <cstring>
#include <ctime>
#include <sys/stat.h>

// ============================================================
// Stage name mapping
// ============================================================

static const char* stage_names[] = {
    "post-load",
    "post-subdivide",
    "post-hierarchy",
    "post-orient",
    "post-position",
    "post-extract",
};

PipelineStage stage_from_name(const char* name) {
    for (int i = 0; i < STAGE_COUNT; ++i)
        if (strcmp(name, stage_names[i]) == 0) return (PipelineStage)i;
    return STAGE_NONE;
}

const char* stage_name(PipelineStage s) {
    if (s >= 0 && s < STAGE_COUNT) return stage_names[s];
    return "unknown";
}

void list_stages() {
    cout << "Available pipeline stages:" << endl;
    for (int i = 0; i < STAGE_COUNT; ++i)
        cout << "  " << i << ". " << stage_names[i] << endl;
}

// ============================================================
// Serialization helpers
// ============================================================

template <typename T>
static void write_scalar(FILE* fp, const T& val) {
    fwrite(&val, sizeof(T), 1, fp);
}

template <typename T>
static void read_scalar(FILE* fp, T& val) {
    size_t r = fread(&val, sizeof(T), 1, fp);
    (void)r;
}

// Write Eigen matrix (rows, cols, data)
template <typename Derived>
static void write_matrix(FILE* fp, const Eigen::MatrixBase<Derived>& mat) {
    uint32_t rows = (uint32_t)mat.rows();
    uint32_t cols = (uint32_t)mat.cols();
    fwrite(&rows, sizeof(uint32_t), 1, fp);
    fwrite(&cols, sizeof(uint32_t), 1, fp);
    if (rows > 0 && cols > 0)
        fwrite(mat.derived().data(), sizeof(typename Derived::Scalar), rows * cols, fp);
}

template <typename MatrixType>
static void read_matrix(FILE* fp, MatrixType& mat) {
    uint32_t rows, cols;
    size_t r;
    r = fread(&rows, sizeof(uint32_t), 1, fp);
    r = fread(&cols, sizeof(uint32_t), 1, fp);
    (void)r;
    mat.resize(rows, cols);
    if (rows > 0 && cols > 0)
        r = fread(mat.data(), sizeof(typename MatrixType::Scalar), rows * cols, fp);
}

// Write adjacency matrix as flat CSR
static void write_adjacency(FILE* fp, const AdjacencyMatrix& adj, uint32_t nVerts) {
    // Count total links
    uint32_t totalLinks = (uint32_t)(adj[nVerts] - adj[0]);
    write_scalar(fp, nVerts);
    write_scalar(fp, totalLinks);

    // Write row offsets
    for (uint32_t i = 0; i <= nVerts; ++i) {
        uint32_t offset = (uint32_t)(adj[i] - adj[0]);
        fwrite(&offset, sizeof(uint32_t), 1, fp);
    }

    // Write link data (id, weight, ivar_uint32)
    for (uint32_t i = 0; i < totalLinks; ++i) {
        const Link& l = adj[0][i];
        fwrite(&l.id, sizeof(uint32_t), 1, fp);
        fwrite(&l.weight, sizeof(float), 1, fp);
        fwrite(&l.ivar_uint32, sizeof(uint32_t), 1, fp);
    }
}

static AdjacencyMatrix read_adjacency(FILE* fp) {
    uint32_t nVerts, totalLinks;
    read_scalar(fp, nVerts);
    read_scalar(fp, totalLinks);

    // Read row offsets
    std::vector<uint32_t> offsets(nVerts + 1);
    size_t r = fread(offsets.data(), sizeof(uint32_t), nVerts + 1, fp);
    (void)r;

    // Allocate
    AdjacencyMatrix adj = new Link*[nVerts + 1];
    Link* links = new Link[totalLinks];

    for (uint32_t i = 0; i <= nVerts; ++i)
        adj[i] = links + offsets[i];

    // Read link data
    for (uint32_t i = 0; i < totalLinks; ++i) {
        r = fread(&links[i].id, sizeof(uint32_t), 1, fp);
        r = fread(&links[i].weight, sizeof(float), 1, fp);
        r = fread(&links[i].ivar_uint32, sizeof(uint32_t), 1, fp);
    }

    return adj;
}

// Write phases
static void write_phases(FILE* fp, const std::vector<std::vector<uint32_t>>& phases) {
    uint32_t nPhases = (uint32_t)phases.size();
    write_scalar(fp, nPhases);
    for (const auto& phase : phases) {
        uint32_t n = (uint32_t)phase.size();
        write_scalar(fp, n);
        if (n > 0)
            fwrite(phase.data(), sizeof(uint32_t), n, fp);
    }
}

static void read_phases(FILE* fp, std::vector<std::vector<uint32_t>>& phases) {
    uint32_t nPhases;
    read_scalar(fp, nPhases);
    phases.resize(nPhases);
    for (auto& phase : phases) {
        uint32_t n;
        read_scalar(fp, n);
        phase.resize(n);
        if (n > 0) {
            size_t r = fread(phase.data(), sizeof(uint32_t), n, fp);
            (void)r;
        }
    }
}

// ============================================================
// Checkpoint file path
// ============================================================

static std::string checkpoint_path(const char* dir, PipelineStage stage) {
    return std::string(dir) + "/" + stage_names[stage] + ".imc";
}

bool checkpoint_exists(const char* dir, PipelineStage stage) {
    struct stat st;
    std::string path = checkpoint_path(dir, stage);
    return stat(path.c_str(), &st) == 0;
}

// ============================================================
// Save checkpoint
// ============================================================

void save_checkpoint(const MultiResolutionHierarchy &mRes, PipelineStage stage,
                     const char* dir, const CheckpointHeader &params) {
    // Ensure directory exists
    mkdir(dir, 0755);

    std::string path = checkpoint_path(dir, stage);
    FILE* fp = fopen(path.c_str(), "wb");
    if (!fp) {
        cout << "[CHECKPOINT] ERROR: Cannot open " << path << " for writing" << endl;
        return;
    }

    Timer<> timer;

    // Write header
    CheckpointHeader hdr = params;
    memset(&hdr.magic, 0, 4);
    memcpy(hdr.magic, "IMC", 4);
    hdr.version = 1;
    strncpy(hdr.stage, stage_names[stage], sizeof(hdr.stage) - 1);
    hdr.stage[sizeof(hdr.stage) - 1] = '\0';
    hdr.timestamp = (long long)time(nullptr);
    fwrite(&hdr, sizeof(hdr), 1, fp);

    // Write stage index
    int stage_idx = (int)stage;
    write_scalar(fp, stage_idx);

    // ---- Always save: hierarchy metadata ----
    int nLevels = mRes.levels();
    write_scalar(fp, nLevels);
    write_scalar(fp, mRes.mScale);
    write_scalar(fp, mRes.mIterationsQ);
    write_scalar(fp, mRes.mIterationsO);
    write_scalar(fp, mRes.mTotalSize);
    write_scalar(fp, mRes.mFrozenQ);
    write_scalar(fp, mRes.mFrozenO);

    // ---- Always save: mesh topology ----
    write_matrix(fp, mRes.mF);
    write_matrix(fp, mRes.mE2E);

    // ---- Save per-level data ----
    for (int i = 0; i < nLevels; ++i) {
        write_matrix(fp, mRes.mV[i]);
        write_matrix(fp, mRes.mN[i]);
        write_matrix(fp, mRes.mA[i]);

        // Adjacency
        write_adjacency(fp, mRes.mAdj[i], mRes.mV[i].cols());

        // Phases
        write_phases(fp, mRes.mPhases[i]);

        // Tree connectivity (not for last level)
        if (i < nLevels - 1) {
            write_matrix(fp, mRes.mToUpper[i]);
            write_matrix(fp, mRes.mToLower[i]);
        }

        // Constraints
        write_matrix(fp, mRes.mCQ[i]);
        write_matrix(fp, mRes.mCO[i]);
        write_matrix(fp, mRes.mCQw[i]);
        write_matrix(fp, mRes.mCOw[i]);
    }

    // ---- Stage-dependent: field data (after post-hierarchy) ----
    if (stage >= STAGE_POST_HIERARCHY) {
        for (int i = 0; i < nLevels; ++i) {
            write_matrix(fp, mRes.mQ[i]);
            write_matrix(fp, mRes.mO[i]);
        }
    }

    fclose(fp);

    // Print summary
    long file_size = 0;
    struct stat st;
    if (stat(path.c_str(), &st) == 0) file_size = st.st_size;
    cout << "[CHECKPOINT] Saved '" << stage_names[stage] << "' to " << path
         << " (" << std::fixed << std::setprecision(1)
         << file_size / (1024.0 * 1024.0) << " MB, took "
         << timeString(timer.value()) << ")" << endl;
}

// ============================================================
// Load checkpoint
// ============================================================

PipelineStage load_checkpoint(MultiResolutionHierarchy &mRes, const char* dir,
                              PipelineStage stage) {
    std::string path = checkpoint_path(dir, stage);
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) {
        cout << "[CHECKPOINT] ERROR: Cannot open " << path << " for reading" << endl;
        return STAGE_NONE;
    }

    Timer<> timer;

    // Read header
    CheckpointHeader hdr;
    size_t r = fread(&hdr, sizeof(hdr), 1, fp);
    (void)r;
    if (memcmp(hdr.magic, "IMC", 4) != 0) {
        cout << "[CHECKPOINT] ERROR: Invalid magic in " << path << endl;
        fclose(fp);
        return STAGE_NONE;
    }
    if (hdr.version != 1) {
        cout << "[CHECKPOINT] ERROR: Unsupported version " << hdr.version << endl;
        fclose(fp);
        return STAGE_NONE;
    }

    int stage_idx;
    read_scalar(fp, stage_idx);
    PipelineStage saved_stage = (PipelineStage)stage_idx;

    cout << "[CHECKPOINT] Loading '" << stage_names[saved_stage] << "' from " << path << endl;
    cout << "[CHECKPOINT]   rosy=" << hdr.rosy << " posy=" << hdr.posy
         << " faces=" << hdr.face_count << " scale=" << hdr.scale << endl;
    cout << "[CHECKPOINT]   strategies: optim=" << hdr.optim_strategy
         << " hier=" << hdr.hierarchy_strategy
         << " extract=" << hdr.extract_strategy << endl;

    // Free existing state
    mRes.free();

    // Read hierarchy metadata
    int nLevels;
    read_scalar(fp, nLevels);
    read_scalar(fp, mRes.mScale);
    read_scalar(fp, mRes.mIterationsQ);
    read_scalar(fp, mRes.mIterationsO);
    read_scalar(fp, mRes.mTotalSize);
    read_scalar(fp, mRes.mFrozenQ);
    read_scalar(fp, mRes.mFrozenO);

    // Read mesh topology
    read_matrix(fp, mRes.mF);
    read_matrix(fp, mRes.mE2E);

    // Read per-level data
    mRes.mV.resize(nLevels);
    mRes.mN.resize(nLevels);
    mRes.mA.resize(nLevels);
    mRes.mAdj.resize(nLevels);
    mRes.mPhases.resize(nLevels);
    mRes.mToUpper.resize(nLevels - 1);
    mRes.mToLower.resize(nLevels - 1);
    mRes.mCQ.resize(nLevels);
    mRes.mCO.resize(nLevels);
    mRes.mCQw.resize(nLevels);
    mRes.mCOw.resize(nLevels);
    mRes.mQ.resize(nLevels);
    mRes.mO.resize(nLevels);

    for (int i = 0; i < nLevels; ++i) {
        read_matrix(fp, mRes.mV[i]);
        read_matrix(fp, mRes.mN[i]);
        read_matrix(fp, mRes.mA[i]);

        mRes.mAdj[i] = read_adjacency(fp);
        read_phases(fp, mRes.mPhases[i]);

        if (i < nLevels - 1) {
            read_matrix(fp, mRes.mToUpper[i]);
            read_matrix(fp, mRes.mToLower[i]);
        }

        read_matrix(fp, mRes.mCQ[i]);
        read_matrix(fp, mRes.mCO[i]);
        read_matrix(fp, mRes.mCQw[i]);
        read_matrix(fp, mRes.mCOw[i]);
    }

    // Stage-dependent: field data
    if (saved_stage >= STAGE_POST_HIERARCHY) {
        for (int i = 0; i < nLevels; ++i) {
            read_matrix(fp, mRes.mQ[i]);
            read_matrix(fp, mRes.mO[i]);
        }
    }

    fclose(fp);
    cout << "[CHECKPOINT] Loaded (took " << timeString(timer.value()) << ")" << endl;
    return saved_stage;
}
