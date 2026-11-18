#!/usr/bin/env bash
# Sets up everything needed to build and run GigaLearnCPP on Linux, then builds it.
# Safe to run again: steps that are already done are skipped.
#
# Usage:   ./setup_linux.sh
#
# Optional environment variables:
#   DEPS_DIR       Where libtorch is downloaded to                      (default: /opt/deps)
#   BUILD_DIR      CMake build folder                                   (default: ./build)
#   JOBS           Parallel compile jobs. Each one needs up to 1.5 GB of memory
#                  (default: as many as fit into the free memory, at most one per CPU thread)
#   ADD_SWAP=0     Never add swap space. By default, if there is too little memory to compile and "sudo" works
#                  without a password, a temporary 4 GB swap file is used for the build and removed afterwards
#   LIBTORCH_URL   Which libtorch to download. The default is the CPU build. For an NVIDIA GPU use e.g.
#                  https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu121.zip
#                  (2.5 GB, the CUDA build was NOT tested by this script's author) together with a fresh DEPS_DIR
#   SKIP_APT=1     Don't try to install system packages
#   PY_INCLUDE / PY_LIBRARY  Development headers/shared library for the selected Python (usually auto-detected)
#   GGL_CXX_FLAGS  Additional flags for *all* C++ targets (e.g. the ABI of a staged libtorch)
#   CONFIGURE_ONLY=1  Prepare dependencies and generate the build, but do not compile
#
# Tested on Debian 13 with GCC 14.2, CMake 3.31, Python 3.13 and the CPU build of libtorch 2.5.1.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
DEPS_DIR="${DEPS_DIR:-/opt/deps}"
BUILD_DIR="${BUILD_DIR:-$REPO_DIR/build}"
JOBS="${JOBS:-}"
SWAP_FILE="${SWAP_FILE:-/swapfile.gigalearn-build}"
SWAP_ADDED=0
TMP_DIR=""
LIBTORCH_URL="${LIBTORCH_URL:-https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcpu.zip}"

# The 16 soccar arena collision meshes that RocketSim needs.
# RocketSim's official way to get them is dumping them from your own copy of the game
# (https://github.com/ZealanL/RLArenaCollisionDumper, Windows only).
# Here they are taken from the source package of rlgym-rocket-league on PyPI, which ships the same files.
MESHES_URL="https://files.pythonhosted.org/packages/99/21/b87ef7ebb0d76e654753349a4f343068d08e1a9eaffd9c77c0593b43668f/rlgym-rocket-league-2.0.1.tar.gz"
MESHES_SHA256="3be8e9d2f1cfb6514e0e3d455fb33ff95e26eb7c399f819fff0258025cf4d501"

step() { printf '\n==> %s\n' "$*"; }

run_as_root() {
	if [ "$(id -u)" = "0" ]; then "$@"; else sudo "$@"; fi
}

can_be_root() {
	[ "$(id -u)" = "0" ] || sudo -n true 2> /dev/null
}

# Free RAM plus free swap, in MB
free_memory_mb() {
	awk '/^MemAvailable:|^SwapFree:/ { sum += $2 } END { printf "%d", sum / 1024 }' /proc/meminfo
}

cleanup() {
	if [ -n "$TMP_DIR" ]; then rm -rf "$TMP_DIR"; fi
	if [ "$SWAP_ADDED" = "1" ]; then
		echo "Removing the temporary swap file $SWAP_FILE."
		run_as_root swapoff "$SWAP_FILE" || true
		run_as_root rm -f "$SWAP_FILE"
	fi
}
trap cleanup EXIT

# ---------------------------------------------------------------------------------------------
step "System packages"
missing=()
for tool in cmake ninja g++ curl unzip; do
	command -v "$tool" > /dev/null || missing+=("$tool")
done
if [ ${#missing[@]} -eq 0 ]; then
	echo "Already installed."
elif [ "${SKIP_APT:-0}" = "1" ] || ! command -v apt-get > /dev/null; then
	echo "Missing tools: ${missing[*]}"
	echo "Please install them (CMake, Ninja, a C++20 compiler, curl, unzip) and run this script again."
	exit 1
else
	echo "Installing (missing: ${missing[*]})..."
	run_as_root apt-get update
	run_as_root apt-get install -y cmake ninja-build g++ curl unzip
fi

# ---------------------------------------------------------------------------------------------
step "Python (GigaLearnCPP embeds a Python interpreter for its metrics)"
PYTHON="${PYTHON:-$(command -v python3 || true)}"
if [ -z "$PYTHON" ]; then
	echo "python3 was not found. Install Python 3 with its development files (Debian/Ubuntu: sudo apt-get install python3-dev)."
	exit 1
fi
PY_INCLUDE="${PY_INCLUDE:-$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')}"
PY_LIBRARY="${PY_LIBRARY:-$("$PYTHON" -c 'import os, sysconfig; print(os.path.join(sysconfig.get_config_var("LIBDIR"), sysconfig.get_config_var("LDLIBRARY")))')}"
echo "Interpreter: $PYTHON ($("$PYTHON" --version 2>&1))"
echo "Headers:     $PY_INCLUDE"
echo "Library:     $PY_LIBRARY"
if [ ! -f "$PY_INCLUDE/Python.h" ] || [ ! -f "$PY_LIBRARY" ] || [[ "$PY_LIBRARY" != *.so* ]]; then
	echo
	echo "This Python has no development headers or no shared library (libpython3.x.so)."
	echo "On Debian/Ubuntu install them with:  sudo apt-get install python3-dev"
	echo "Or point this script at another interpreter with:  PYTHON=/path/to/python3 ./setup_linux.sh"
	exit 1
fi

# ---------------------------------------------------------------------------------------------
step "libtorch"
if [ -f "$DEPS_DIR/libtorch/share/cmake/Torch/TorchConfig.cmake" ]; then
	echo "Already at $DEPS_DIR/libtorch (version $(cat "$DEPS_DIR/libtorch/build-version" 2> /dev/null || echo unknown))."
else
	if ! mkdir -p "$DEPS_DIR" 2> /dev/null || [ ! -w "$DEPS_DIR" ]; then
		run_as_root mkdir -p "$DEPS_DIR"
		run_as_root chown "$(id -u):$(id -g)" "$DEPS_DIR"
	fi
	echo "Downloading $LIBTORCH_URL ..."
	curl -L --fail --progress-bar -o "$DEPS_DIR/libtorch.zip" "$LIBTORCH_URL"
	echo "Unpacking..."
	unzip -q -o "$DEPS_DIR/libtorch.zip" -d "$DEPS_DIR"
	rm -f "$DEPS_DIR/libtorch.zip"
	echo "Installed libtorch $(cat "$DEPS_DIR/libtorch/build-version") to $DEPS_DIR/libtorch."
fi

# ---------------------------------------------------------------------------------------------
step "RocketSim collision meshes"
if [ -f "$REPO_DIR/collision_meshes/soccar/mesh_15.cmf" ]; then
	echo "Already at $REPO_DIR/collision_meshes."
else
	TMP_DIR="$(mktemp -d)"
	tmp_dir="$TMP_DIR"
	echo "Downloading $MESHES_URL ..."
	curl -L --fail -sS -o "$tmp_dir/rlgym.tar.gz" "$MESHES_URL"
	echo "$MESHES_SHA256  $tmp_dir/rlgym.tar.gz" | sha256sum -c -
	tar -xzf "$tmp_dir/rlgym.tar.gz" -C "$tmp_dir"
	mkdir -p "$REPO_DIR/collision_meshes"
	cp -r "$tmp_dir"/rlgym-rocket-league-*/rlgym/rocket_league/sim/collision_meshes/soccar "$REPO_DIR/collision_meshes/"
	echo "Installed $(ls "$REPO_DIR/collision_meshes/soccar" | wc -l) meshes to $REPO_DIR/collision_meshes/soccar."
fi

# ---------------------------------------------------------------------------------------------
step "Configuring (build folder: $BUILD_DIR)"
cmake -S "$REPO_DIR" -B "$BUILD_DIR" -G Ninja -Wno-dev \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_PREFIX_PATH="$DEPS_DIR/libtorch" \
	-DCMAKE_CXX_FLAGS="-march=native${GGL_CXX_FLAGS:+ $GGL_CXX_FLAGS}" \
	-DPython_EXECUTABLE="$PYTHON" \
	-DPYTHON_EXECUTABLE="$PYTHON" \
	-DPython_INCLUDE_DIR="$PY_INCLUDE" -DPYTHON_INCLUDE_DIR="$PY_INCLUDE" \
	-DPython_LIBRARY="$PY_LIBRARY" -DPYTHON_LIBRARY="$PY_LIBRARY"

if [ "${CONFIGURE_ONLY:-0}" = 1 ]; then exit 0; fi

# ---------------------------------------------------------------------------------------------
step "Memory for compiling"
# Measured with GCC 14.2 and -O3: compiling one of the files that include libtorch takes up to 1.5 GB of RAM
# (Learner.cpp 1463 MB, PolicyVersionManager.cpp 1400 MB, PPOLearner.cpp 1166 MB, ...).
# With less memory than that the machine does not fail cleanly, it swaps itself to a standstill. So check first.
MB_PER_JOB=1600
if [ -r /proc/meminfo ]; then
	free_mb="$(free_memory_mb)"
	echo "Free memory (RAM + swap): $free_mb MB. One compile job needs up to about $MB_PER_JOB MB."
	if [ "$free_mb" -lt $((MB_PER_JOB + 400)) ]; then
		disk_free_mb="$(df --output=avail -m "$(dirname "$SWAP_FILE")" 2> /dev/null | tail -1 | tr -d ' ' || echo 0)"
		if [ "${ADD_SWAP:-1}" = "1" ] && can_be_root && [ "${disk_free_mb:-0}" -gt 6000 ]; then
			echo "That is too little. Adding a temporary 4 GB swap file ($SWAP_FILE) for the build."
			run_as_root rm -f "$SWAP_FILE"
			if run_as_root fallocate -l 4G "$SWAP_FILE" && run_as_root chmod 600 "$SWAP_FILE" \
				&& run_as_root mkswap "$SWAP_FILE" > /dev/null && run_as_root swapon "$SWAP_FILE"; then
				SWAP_ADDED=1
			else
				run_as_root rm -f "$SWAP_FILE"
				echo "Could not add the swap file. Please add at least 3 GB of swap space yourself and run this script again."
				exit 1
			fi
			free_mb="$(free_memory_mb)"
		elif [ "${FORCE_BUILD:-0}" = "1" ]; then
			echo "That is too little, but FORCE_BUILD=1 is set. Trying anyway."
		else
			echo "That is too little to compile. Add swap space and run this script again, for example:"
			echo "  sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile"
			echo "(Set FORCE_BUILD=1 to try anyway.)"
			exit 1
		fi
	fi
	if [ -z "$JOBS" ]; then
		JOBS=$((free_mb / MB_PER_JOB))
		cpu_threads="$(nproc 2> /dev/null || echo 1)"
		if [ "$JOBS" -gt "$cpu_threads" ]; then JOBS="$cpu_threads"; fi
		if [ "$JOBS" -lt 1 ]; then JOBS=1; fi
	fi
fi
JOBS="${JOBS:-1}"

# ---------------------------------------------------------------------------------------------
step "Building with $JOBS parallel jobs"
cmake --build "$BUILD_DIR" --target GigaLearnBot -j "$JOBS"

step "Done"
echo "Executable: $BUILD_DIR/GigaLearnBot"
echo
echo "Resume paused CPU training: python3 tools/resume_training.py"
echo "Train on a GPU:      $BUILD_DIR/GigaLearnBot            (defaults = the original example's settings)"
echo "Evaluate:            $BUILD_DIR/GigaLearnBot --eval checkpoints"
echo "All options:         $BUILD_DIR/GigaLearnBot --help"
