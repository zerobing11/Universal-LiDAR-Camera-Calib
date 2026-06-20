#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
INSTALL_DIR="${SCRIPT_DIR}/sqpnp-install"
SOURCE_DIR="${SCRIPT_DIR}/sqpnp"

if [ ! -d "${SOURCE_DIR}" ]; then
    SOURCE_DIR="${PROJECT_ROOT}/sqpnp"
fi

if [ ! -d "${SOURCE_DIR}" ]; then
    echo "Error: sqpnp source not found."
    echo "Tried:"
    echo "  ${SCRIPT_DIR}/sqpnp"
    echo "  ${PROJECT_ROOT}/sqpnp"
    exit 1
fi

rm -rf "${INSTALL_DIR}"
mkdir -p "${INSTALL_DIR}/build"
cd "${INSTALL_DIR}/build"
cmake "${SOURCE_DIR}" -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}"
make -j10
make install

