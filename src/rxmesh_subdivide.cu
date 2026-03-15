/*
    rxmesh_subdivide.cu — GPU edge splitting using RXMesh cavity operator.
    Compiled separately (scripts/build_rxmesh.sh) to avoid Eigen version
    conflicts and NVCC codegen regression from multiple .cu files.

    Status: functional but slower than CPU for one-shot bulk subdivision.
    RXMesh excels at iterative remeshing (split+collapse+flip+smooth loops),
    not one-shot "split all long edges" which the CPU does in ~500ms via
    priority queue.
*/

#include "rxmesh_subdivide.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_profiler_api.h>
#include "rxmesh/rxmesh_dynamic.h"
#include "rxmesh/cavity_manager.cuh"
#include "rxmesh/kernels/query_kernel.cuh"
#include "rxmesh/util/log.h"

using namespace rxmesh;

template <uint32_t blockThreads>
__global__ static void kernel_edge_split(
    Context context, VertexAttribute<float> coords,
    EdgeAttribute<int8_t> edge_status, const float max_len_sq)
{
    auto block = cooperative_groups::this_thread_block();
    ShmemAllocator shrd_alloc;
    CavityManager<blockThreads, CavityOp::E> cavity(block, context, shrd_alloc, true);
    if (cavity.patch_id() == INVALID32) return;
    uint32_t shmem_before = shrd_alloc.get_allocated_size_bytes();

    auto should_split = [&](const EdgeHandle& eh, const VertexIterator& iter) {
        if (edge_status(eh) != 0) return;

        const VertexHandle va = iter[0], vb = iter[2];
        const VertexHandle vc = iter[1], vd = iter[3];

        if (!vc.is_valid() || !vd.is_valid() || !va.is_valid() || !vb.is_valid()) {
            edge_status(eh) = 1;
            return;
        }

        float dx = coords(va,0)-coords(vb,0);
        float dy = coords(va,1)-coords(vb,1);
        float dz = coords(va,2)-coords(vb,2);
        float len_sq = dx*dx + dy*dy + dz*dz;

        if (len_sq > max_len_sq) {
            cavity.create(eh);
        } else {
            edge_status(eh) = 1;
        }
    };

    Query<blockThreads> query(context, cavity.patch_id());
    query.dispatch<Op::EVDiamond>(block, shrd_alloc, should_split);
    block.sync();
    shrd_alloc.dealloc(shrd_alloc.get_allocated_size_bytes() - shmem_before);

    if (cavity.prologue(block, shrd_alloc, coords, edge_status)) {
        cavity.for_each_cavity(block, [&](uint16_t c, uint16_t size) {
            assert(size == 4);

            const VertexHandle v0 = cavity.get_cavity_vertex(c, 0);
            const VertexHandle v1 = cavity.get_cavity_vertex(c, 2);

            const VertexHandle new_v = cavity.add_vertex();
            if (!new_v.is_valid()) return;

            coords(new_v, 0) = (coords(v0, 0) + coords(v1, 0)) * 0.5f;
            coords(new_v, 1) = (coords(v0, 1) + coords(v1, 1)) * 0.5f;
            coords(new_v, 2) = (coords(v0, 2) + coords(v1, 2)) * 0.5f;

            // Fan triangulation (RXMesh canonical pattern from apps/Remesh/split.cuh)
            DEdgeHandle e0 = cavity.add_edge(new_v, cavity.get_cavity_vertex(c, 0));
            const DEdgeHandle e_init = e0;

            if (e0.is_valid()) {
                for (uint16_t i = 0; i < size; ++i) {
                    const DEdgeHandle e = cavity.get_cavity_edge(c, i);
                    const DEdgeHandle e1 =
                        (i == size - 1) ?
                            e_init.get_flip_dedge() :
                            cavity.add_edge(
                                cavity.get_cavity_vertex(c, i + 1), new_v);
                    if (!e1.is_valid()) break;

                    const FaceHandle f = cavity.add_face(e0, e, e1);
                    if (!f.is_valid()) break;
                    e0 = e1.get_flip_dedge();
                }
            }
        });
    }

    cavity.epilogue(block);
    block.sync();
}

extern "C" void rxmesh_subdivide(
    const float* V_in, uint32_t nV_in, const uint32_t* F_in, uint32_t nF_in,
    float maxLength, float** V_out, uint32_t* nV_out, uint32_t** F_out, uint32_t* nF_out)
{
    static bool log_initialized = false;
    if (!log_initialized) {
        rxmesh::Log::init(spdlog::level::warn);
        log_initialized = true;
    }

    float max_len_sq = maxLength * maxLength;
    char tmp_in[256], tmp_out[256];
    snprintf(tmp_in, sizeof(tmp_in), "/tmp/im_rx_%d.obj", (int)getpid());
    snprintf(tmp_out, sizeof(tmp_out), "/tmp/im_rx_out_%d.obj", (int)getpid());

    // Write input mesh as OBJ for RXMesh
    FILE *fp = fopen(tmp_in, "w");
    for (uint32_t v = 0; v < nV_in; v++)
        fprintf(fp, "v %.15g %.15g %.15g\n", V_in[v*3], V_in[v*3+1], V_in[v*3+2]);
    for (uint32_t f = 0; f < nF_in; f++)
        fprintf(fp, "f %u %u %u\n", F_in[f*3]+1, F_in[f*3+1]+1, F_in[f*3+2]+1);
    fclose(fp);

    // patch_size=256, over_alloc=4.0, num_sec=4 (boffins session recommendations)
    RXMeshDynamic rx(std::string(tmp_in), "", 256, 4.0, 4);
    auto coords = rx.get_input_vertex_coordinates();
    auto edge_status = rx.add_edge_attribute<int8_t>("st", 1);
    constexpr uint32_t BT = 256;

    fprintf(stderr, "[RXMESH] Init: %u verts, %u edges, %u faces, %u patches\n",
            rx.get_num_vertices(), rx.get_num_edges(), rx.get_num_faces(),
            rx.get_num_patches());

    uint32_t prev_v = rx.get_num_vertices(true);

    for (int it = 0; it < 30; it++) {
        edge_status->reset(int8_t(0), DEVICE);
        rx.reset_scheduler();

        LaunchBox<BT> lb;
        rx.update_launch_box({Op::EVDiamond}, lb,
                             (void*)kernel_edge_split<BT>, true);

        int inner = 0;
        while (!rx.is_queue_empty()) {
            kernel_edge_split<BT><<<lb.blocks, lb.num_threads, lb.smem_bytes_dyn>>>(
                rx.get_context(), *coords, *edge_status, max_len_sq);

            cudaError_t err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                fprintf(stderr, "[RXMESH] CUDA error: %s\n", cudaGetErrorString(err));
                break;
            }

            rx.cleanup();
            rx.slice_patches(*coords, *edge_status);
            rx.cleanup();
            inner++;
        }

        uint32_t cur_v = rx.get_num_vertices(true);
        fprintf(stderr, "[RXMESH] iter %d (%d inner): %u->%u verts\n",
                it, inner, prev_v, cur_v);
        if (cur_v == prev_v) break;
        prev_v = cur_v;
    }

    // Export result
    rx.export_obj(tmp_out, *coords);
    fp = fopen(tmp_out, "r");
    std::vector<float> verts;
    std::vector<uint32_t> faces;
    char line[256];
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == 'v' && line[1] == ' ') {
            float x, y, z;
            sscanf(line + 2, "%f %f %f", &x, &y, &z);
            verts.push_back(x); verts.push_back(y); verts.push_back(z);
        } else if (line[0] == 'f' && line[1] == ' ') {
            uint32_t a, b, c;
            sscanf(line + 2, "%u %u %u", &a, &b, &c);
            faces.push_back(a-1); faces.push_back(b-1); faces.push_back(c-1);
        }
    }
    fclose(fp);

    *nV_out = (uint32_t)(verts.size() / 3);
    *nF_out = (uint32_t)(faces.size() / 3);
    *V_out = (float*)malloc(verts.size() * sizeof(float));
    *F_out = (uint32_t*)malloc(faces.size() * sizeof(uint32_t));
    memcpy(*V_out, verts.data(), verts.size() * sizeof(float));
    memcpy(*F_out, faces.data(), faces.size() * sizeof(uint32_t));
    remove(tmp_in);
    remove(tmp_out);
    fprintf(stderr, "[RXMESH] Done: %u verts, %u faces\n", *nV_out, *nF_out);
}
