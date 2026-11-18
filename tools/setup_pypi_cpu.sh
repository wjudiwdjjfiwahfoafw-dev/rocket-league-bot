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
if [ ! -x "$cache/tools/cmake/data/bin/cmake" ] || [ ! -x "$cache/tools/bin/ninja" ]; then
    "${pip_base[@]}" install "${index[@]}" --upgrade --target "$cache/tools" cmake==3.31.6 ninja==1.11.1.4
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
for archive, members in {
    'torch-2.5.1-cp311-cp311-manylinux1_x86_64.whl': ('torch/include/', 'torch/lib/libtorch_cpu.so', 'torch/lib/libc10.so', 'torch/lib/libgomp-'),
    'nvidia_cuda_cupti_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl': ('nvidia/cuda_cupti/lib/libcupti.so.12',),
    'nvidia_cuda_runtime_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl': ('nvidia/cuda_runtime/lib/libcudart.so.12',),
}.items():
    with zipfile.ZipFile(root / 'wheels' / archive) as z:
        for member in z.infolist():
            if not member.is_dir() and member.filename.startswith(members):
                name = member.filename
                dest = root / ('libtorch/' + name[6:] if name.startswith('torch/') else name)
                dest.parent.mkdir(parents=True, exist_ok=True)
                with z.open(member) as src, dest.open('wb') as dst:
                    shutil.copyfileobj(src, dst)
PY
    cat > "$config" <<'CMAKE'
get_filename_component(TORCH_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
add_library(torch SHARED IMPORTED)
set_target_properties(torch PROPERTIES IMPORTED_LOCATION "${TORCH_INSTALL_PREFIX}/lib/libtorch_cpu.so" INTERFACE_INCLUDE_DIRECTORIES "${TORCH_INSTALL_PREFIX}/include;${TORCH_INSTALL_PREFIX}/include/torch/csrc/api/include" INTERFACE_LINK_LIBRARIES "${TORCH_INSTALL_PREFIX}/lib/libc10.so")
set(TORCH_LIBRARIES torch)
set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")
CMAKE
fi

# Prefer installed dev files. The minimal Debian 12 sandbox has only the interpreter.
include="${PY_INCLUDE:-$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_path("include"))')}"
library="${PY_LIBRARY:-$("$PYTHON" -c 'import os,sysconfig as s; print(os.path.join(s.get_config_var("LIBDIR"), s.get_config_var("LDLIBRARY")))')}"
if [ ! -f "$include/Python.h" ] || [ ! -f "$library" ]; then
    if [ "$("$PYTHON" -c 'import platform; print(platform.python_version())')" != 3.11.2 ]; then
        echo 'Please install matching python3-dev or supply PY_INCLUDE / PY_LIBRARY. The source fallback matches 3.11.2 only.' >&2
        exit 1
    fi
    include="$cache/python/include"; library="$cache/python/lib/libpython3.11.so"
    if [ ! -f "$include/pyconfig.h" ] || [ ! -f "$library" ] || [ ! -f "$cache/python/render-stdlib-v1" ]; then
        mkdir -p "$cache/cpython" "$include" "$(dirname "$library")"
        (
            cd "$cache/cpython"
            curl -fsSL -o source.tgz https://codeload.github.com/python/cpython/tar.gz/refs/tags/v3.11.2
            echo '5ce2b6e8363d3a170e6fc2c5b7ef26f2c2565d4c4183ff092d3d36b7727499e8  source.tgz' | sha256sum -c -
            tar -xzf source.tgz --strip-components=1
            # Match built-ins needed by metrics and the UDP render receiver (Debian builds these into its executable).
            printf '*static*\nmath mathmodule.c -lm\n_socket socketmodule.c\nselect selectmodule.c\narray arraymodule.c\n_struct _struct.c\n' > Modules/Setup.local
            ./configure --prefix="$("$PYTHON" -c 'import sys; print(sys.base_prefix)')" --enable-shared --without-ensurepip CFLAGS='-O1 -g0'
            make -j "${JOBS:-2}" libpython3.11.so
            cp -a Include/. "$include/"; cp pyconfig.h "$include/"
            cp -a libpython3.11.so* "$(dirname "$library")/"
            touch "$cache/python/render-stdlib-v1"
        )
    fi
fi
# Reuse the standard Release build and mesh installer; don't change the host or install Python packages globally.
export PYTHON PY_INCLUDE="$include" PY_LIBRARY="$library" DEPS_DIR="$cache" SKIP_APT=1 ADD_SWAP=0
export GGL_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"
exec bash ./setup_linux.sh
