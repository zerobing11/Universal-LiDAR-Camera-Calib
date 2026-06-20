#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
INSTALL_DIR="${SCRIPT_DIR}/yaml-cpp-install"
SOURCE_DIR="${SCRIPT_DIR}/yaml-cpp"

if [ ! -d "${SOURCE_DIR}" ]; then
    SOURCE_DIR="${PROJECT_ROOT}/yaml-cpp"
fi

if [ ! -d "${SOURCE_DIR}" ]; then
    echo "Error: yaml-cpp source not found."
    echo "Tried:"
    echo "  ${SCRIPT_DIR}/yaml-cpp"
    echo "  ${PROJECT_ROOT}/yaml-cpp"
    exit 1
fi

rm -rf "${INSTALL_DIR}"
mkdir -p "${INSTALL_DIR}/build"
cd "${INSTALL_DIR}/build"
cmake "${SOURCE_DIR}" -DYAML_BUILD_SHARED_LIBS=ON \
                      -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}"
make -j10
make install

CONFIG_DIR="${INSTALL_DIR}/lib/cmake/yaml-cpp"
if [ -f "${CONFIG_DIR}/yaml-cpp-config.cmake" ]; then
    mv "${CONFIG_DIR}/yaml-cpp-config.cmake" "${CONFIG_DIR}/YAML_CPPConfig.cmake"
fi
