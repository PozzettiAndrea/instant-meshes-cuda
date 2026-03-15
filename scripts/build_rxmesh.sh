#!/usr/bin/env bash
# Build RXMesh static library + rxmesh_subdivide device-linked objects
# for linking into Instant Meshes.
#
# Usage: scripts/build_rxmesh.sh [CUDA_PATH] [ARCH]
#   CUDA_PATH defaults to /usr/local/cuda-13.0
#   ARCH defaults to 86 (sm_86)
#
# Output: build/rxmesh/
#   libRXMesh.a, libmetis.a, libGKlib.a
#   rxmesh_subdivide.o, rxmesh_dlink.o

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CUDA_PATH="${1:-/usr/local/cuda-13.0}"
ARCH="${2:-86}"

NVCC="$CUDA_PATH/bin/nvcc"
if [ ! -x "$NVCC" ]; then
    echo "ERROR: nvcc not found at $NVCC" >&2
    exit 1
fi

# Find a g++-11 or fall back to default
HOST_CXX="$(which g++-11 2>/dev/null || which g++ 2>/dev/null)"
echo "Using CUDA: $NVCC ($(${NVCC} --version | grep release | sed 's/.*release //' | sed 's/,.*//'))"
echo "Using host compiler: $HOST_CXX"
echo "Target arch: sm_$ARCH"

RXMESH_SRC="$ROOT_DIR/ext/rxmesh"
BUILD_DIR="$ROOT_DIR/build/rxmesh"
mkdir -p "$BUILD_DIR"

# Step 1: Build libRXMesh.a via CMake (if not already built)
if [ ! -f "$BUILD_DIR/libRXMesh.a" ]; then
    echo "=== Building libRXMesh.a ==="
    CMAKE_BUILD="$BUILD_DIR/cmake_build"
    mkdir -p "$CMAKE_BUILD"
    cd "$CMAKE_BUILD"
    cmake "$RXMESH_SRC" \
        -DCMAKE_CUDA_COMPILER="$NVCC" \
        -DCMAKE_CUDA_HOST_COMPILER="$HOST_CXX" \
        -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
        -DCMAKE_BUILD_TYPE=Release \
        -DRX_USE_POLYSCOPE=OFF \
        -DRX_BUILD_TESTS=OFF \
        -DRX_BUILD_APPS=OFF \
        -DRX_USE_SUITESPARSE=OFF \
        -DRX_USE_CUDSS=OFF \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    make -j"$(nproc)" RXMesh

    # Copy artifacts to build/rxmesh/
    cp libRXMesh.a "$BUILD_DIR/"
    cp _deps/metis-build/libmetis.a "$BUILD_DIR/"
    cp _deps/metis-build/GKlib/libGKlib.a "$BUILD_DIR/"

    # Stash include paths for rxmesh_subdivide compilation
    echo "$CMAKE_BUILD/_deps/eigen-src" > "$BUILD_DIR/eigen_include.txt"
    echo "$CMAKE_BUILD/_deps/spdlog-src/include" > "$BUILD_DIR/spdlog_include.txt"
    echo "$CMAKE_BUILD/_deps/cereal-src/include" > "$BUILD_DIR/cereal_include.txt"
    cd "$ROOT_DIR"
    echo "=== libRXMesh.a built ==="
else
    echo "=== libRXMesh.a already exists, skipping ==="
fi

# Step 2: Compile rxmesh_subdivide.cu
EIGEN_INC="$(cat "$BUILD_DIR/eigen_include.txt")"
SPDLOG_INC="$(cat "$BUILD_DIR/spdlog_include.txt")"
CEREAL_INC="$(cat "$BUILD_DIR/cereal_include.txt")"

echo "=== Compiling rxmesh_subdivide.cu ==="
"$NVCC" -c "$ROOT_DIR/src/rxmesh_subdivide.cu" -o "$BUILD_DIR/rxmesh_subdivide.o" \
    -I "$RXMESH_SRC/include" \
    -I "$EIGEN_INC" \
    -I "$SPDLOG_INC" \
    -I "$CEREAL_INC" \
    --std=c++17 -arch=sm_"$ARCH" -O3 -ccbin "$HOST_CXX" \
    --expt-relaxed-constexpr -rdc=true

# Step 3: Device-link rxmesh_subdivide.o + libRXMesh.a
echo "=== Device-linking ==="
"$NVCC" -dlink "$BUILD_DIR/rxmesh_subdivide.o" "$BUILD_DIR/libRXMesh.a" \
    -o "$BUILD_DIR/rxmesh_dlink.o" -arch=sm_"$ARCH" -ccbin "$HOST_CXX"

echo "=== Done ==="
echo "Artifacts in $BUILD_DIR:"
ls -lh "$BUILD_DIR"/*.{a,o} 2>/dev/null
