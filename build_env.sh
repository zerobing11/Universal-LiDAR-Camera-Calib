#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="${ROOT_DIR}/env"

cd "${ENV_DIR}"

echo "==> Building corner-detection..."
bash ./compile_corner-detection.sh

echo "==> Building sqpnp..."
bash ./compile_sqpnp.sh

echo "==> Building yaml-cpp..."
bash ./compile_yaml_cpp.sh

echo "==> All dependencies compiled successfully."
