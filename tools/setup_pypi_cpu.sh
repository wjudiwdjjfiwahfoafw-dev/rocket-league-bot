#!/usr/bin/env bash
# No-root alternative to setup_linux.sh for a Linux x86-64 CPython 3.11 host with GCC, make, curl and pip,
# but no system CMake/Ninja, Python dev package or access to the official libtorch zip. CPU only.
# Pinned wheels total ~880 MiB to download. Only CPU-linkable libraries/headers are unpacked; no GPU driver required.
# Usage: bash tools/setup_pypi_cpu.sh   (WHEEL_DEPS_DIR, BUILD_DIR, JOBS and CONFIGURE_ONLY=1 are optional)
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-$(command -v python3)}"
if ! "$PYTHON" -c 'import platform, sys; sys.exit(not (sys.platform == "linux" and platform.machine() == "x86_64" and sys.version_info[:2] == (3, 11)))'; then
    echo 'This fallback is pinned to Linux x86-64 / Python 3.11. Use setup_linux.sh on other systems.' >&2
    exit 1
fi
cache="${WHEEL_DEPS_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/gigalearn-pypi-cpu-2.5.1}"
mkdir -p "$cache"
cache="$(cd "$cache" && pwd)"
pip_base=("$PYTHON" -m pip --disable-pip-version-check)
index=(--only-binary=:all: --no-deps --index-url https://pypi.org/simple)
if [ ! -x "$cache/tools/cmake/data/bin/cmake" ] || [ ! -x "$cache/tools/bin/ninja" ] || [ ! -x "$cache/tools/bin/patchelf" ]; then
    "${pip_base[@]}" install "${index[@]}" --upgrade --target "$cache/tools" cmake==3.31.6 ninja==1.11.1.4 patchelf==0.19.3.0
fi
export PATH="$cache/tools/cmake/data/bin:$cache/tools/bin:$PATH"

# The PyPI wheel's normal TorchConfig requires CUDA. Here the C++ frontend and CPU ops
# are in libtorch_cpu.so; its dynamic dependencies still include the CUPTI and runtime sonames.
config="$cache/libtorch/share/cmake/Torch/TorchConfig.cmake"
if [ ! -s "$config" ]; then
    mkdir -p "$cache/wheels" "$(dirname "$config")"
    cat > "$cache/wheels/pins.txt" <<'PINS'
torch==2.5.1 --hash=sha256:de5b7d6740c4b636ef4db92be922f0edc425b65ed78c5076c43c42d362a45457
nvidia-cuda-cupti-cu12==12.4.127 --hash=sha256:9dec60f5ac126f7bb551c055072b69d85392b13311fcc1bcda2202d172df30fb
nvidia-cuda-runtime-cu12==12.4.127 --hash=sha256:64403288fa2136ee8e467cdc9c9427e0434110899d07c779f25b5c068934faa5
PINS
    "${pip_base[@]}" download "${index[@]}" --require-hashes --dest "$cache/wheels" -r "$cache/wheels/pins.txt"
    "$PYTHON" - "$cache" <<'PY'
from pathlib import Path
import shutil, sys, zipfile
root = Path(sys.argv[1])
# libtorch_cpu.so lists libcudart.so.12 and libcupti.so.12 as NEEDED, so both have to sit next to it:
# the linker resolves them from there and the loader finds them there at run time.
for archive, members in {
    'torch-2.5.1-cp311-cp311-manylinux1_x86_64.whl': ('torch/include/', 'torch/lib/libtorch_cpu.so', 'torch/lib/libc10.so', 'torch/lib/libgomp-'),
    'nvidia_cuda_cupti_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl': ('nvidia/cuda_cupti/lib/libcupti.so.12',),
    'nvidia_cuda_runtime_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl': ('nvidia/cuda_runtime/lib/libcudart.so.12',),
}.items():
    with zipfile.ZipFile(root / 'wheels' / archive) as z:
        for member in z.infolist():
            if not member.is_dir() and member.filename.startswith(members):
                name = member.filename
                if name.startswith('torch/'):
                    dest = root / ('libtorch/' + name[6:])
                else:
                    dest = root / ('libtorch/lib/' + name.rsplit('/', 1)[-1])
                dest.parent.mkdir(parents=True, exist_ok=True)
                with z.open(member) as src, dest.open('wb') as dst:
                    shutil.copyfileobj(src, dst)
PY
    cat > "$config" <<'CMAKE'
get_filename_component(TORCH_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
add_library(torch SHARED IMPORTED)
set_target_properties(torch PROPERTIES
  IMPORTED_LOCATION "${TORCH_INSTALL_PREFIX}/lib/libtorch_cpu.so"
  INTERFACE_INCLUDE_DIRECTORIES "${TORCH_INSTALL_PREFIX}/include;${TORCH_INSTALL_PREFIX}/include/torch/csrc/api/include"
  INTERFACE_LINK_DIRECTORIES "${TORCH_INSTALL_PREFIX}/lib"
  INTERFACE_LINK_LIBRARIES "${TORCH_INSTALL_PREFIX}/lib/libc10.so;${TORCH_INSTALL_PREFIX}/lib/libcupti.so.12;${TORCH_INSTALL_PREFIX}/lib/libcudart.so.12")
set(TORCH_LIBRARIES torch)
set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")
CMAKE
    # The loader has to find libcupti.so.12/libcudart.so.12 next to libtorch_cpu.so at run time.
    if [ -x "$cache/tools/bin/patchelf" ]; then
        "$cache/tools/bin/patchelf" --set-rpath '$ORIGIN' "$cache/libtorch/lib/libtorch_cpu.so" "$cache/libtorch/lib/libc10.so"
    fi
fi

# The embedded interpreter loads the standard library of the interpreter it is told about, so the
# headers, the shared library, the standard library and the interpreter must all come from the same
# installation. Mixing them (for example Debian's standard library with a self-built libpython) makes
# modules that one of the two builds in as a builtin unavailable to the other.
python_prefix="${PYTHON_PREFIX:-$cache/python}"
include="$python_prefix/include/python3.11"
library="$python_prefix/lib/libpython3.11.so"
interpreter="$python_prefix/bin/python3.11"

# A prefix is only usable when the interpreter, the headers, the shared library and a standard
# library that provides the modules the metrics scripts import all come from that same prefix.
prefix_is_complete() {
    [ -x "$1/bin/python3.11" ] && [ -f "$1/include/python3.11/Python.h" ] && [ -f "$1/lib/libpython3.11.so" ] || return 1
    [ -d "$1/lib/python3.11" ] || return 1
    "$1/bin/python3.11" -c 'import math, json, sys' > /dev/null 2>&1
}

if ! prefix_is_complete "$python_prefix"; then
    # Reuse an already installed, self-consistent CPython before spending time on a source build.
    for candidate in "$cache/python" /home/user/py311 /usr/local; do
        if prefix_is_complete "$candidate"; then
            python_prefix="$candidate"; break
        fi
    done
fi
include="$python_prefix/include/python3.11"
library="$python_prefix/lib/libpython3.11.so"
interpreter="$python_prefix/bin/python3.11"
if ! prefix_is_complete "$python_prefix"; then
    if [ "$("$PYTHON" -c 'import platform; print(platform.python_version())')" != 3.11.2 ]; then
        echo 'Please install matching python3-dev or supply PYTHON_PREFIX with a CPython 3.11.2 tree.' >&2
        exit 1
    fi
    mkdir -p "$cache/cpython"
    (
        cd "$cache/cpython"
        curl -fsSL -o source.tgz https://codeload.github.com/python/cpython/tar.gz/refs/tags/v3.11.2
        echo '5ce2b6e8363d3a170e6fc2c5b7ef26f2c2565d4c4183ff092d3d36b7727499e8  source.tgz' | sha256sum -c -
        tar -xzf source.tgz --strip-components=1
        # Install the whole interpreter, standard library and extension modules under one prefix, so the
        # embedded interpreter and the linked libpython always come from the same build.
        ./configure --prefix="$python_prefix" --enable-shared --without-ensurepip CFLAGS='-O1 -g0'
        make -j "${JOBS:-2}"
        make install
    )
    # The installed interpreter has to find its own libpython without a caller-set library path.
    if [ -x "$cache/tools/bin/patchelf" ]; then
        "$cache/tools/bin/patchelf" --set-rpath '$ORIGIN/../lib' "$interpreter"
    fi
fi
# Reuse the standard Release build and mesh installer; don't change the host or install Python packages globally.
export PYTHON="$interpreter" PY_INCLUDE="$include" PY_LIBRARY="$library" DEPS_DIR="$cache" SKIP_APT=1 ADD_SWAP=0
export GGL_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"
exec bash ./setup_linux.sh
