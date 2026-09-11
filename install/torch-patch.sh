#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# PyTorch library.py patch script
# Fixes shutdown crashes by replacing hasattr check with vars() check
# Usage: ./torch-patch.sh

set -e

echo "=== Patching PyTorch library.py ==="

# Find torch site-packages location
TORCH_SITE_PACKAGES=$(python -c 'from pathlib import Path; import torch.library; print(Path(torch.library.__file__).parent.parent)')
LIBRARY_PY="${TORCH_SITE_PACKAGES}/torch/library.py"

if [[ -f "${LIBRARY_PY}" ]]; then
    # Create backup
    cp "${LIBRARY_PY}" "${LIBRARY_PY}.bak"

    # Replace the problematic line
    sed -i 's/if not hasattr(namespace, name):/if name not in vars(namespace):/' "${LIBRARY_PY}"

    # Verify the change
    if grep -q "if name not in vars(namespace):" "${LIBRARY_PY}"; then
        echo "PyTorch library.py patched successfully"
    else
        echo "Warning: Failed to patch library.py, restoring backup"
        mv "${LIBRARY_PY}.bak" "${LIBRARY_PY}"
    fi
else
    echo "Warning: torch/library.py not found at ${LIBRARY_PY}"
fi
