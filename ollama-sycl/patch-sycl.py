#!/usr/bin/env python3
"""
Patch upstream ggml-sycl to match ollama's modified ggml backend API:
1. graph_compute() had an extra 'int batch_size' parameter (ollama addition)
"""

import re
import sys

path = sys.argv[1]
with open(path, "r") as f:
    src = f.read()

original = src

# 1. Fix graph_compute signature: add 'int batch_size' parameter
if "ggml_backend_sycl_graph_compute" in src and "int batch_size" not in src:
    src = re.sub(
        r'(static\s+(?:enum\s+)?ggml_status\s+ggml_backend_sycl_graph_compute\s*\([^)]*cgraph)\s*\)',
        r'\1, int batch_size)',
        src,
    )
    src = re.sub(
        r'(ggml_backend_sycl_graph_compute\([^)]*int\s+batch_size\)\s*\{)',
        r'\1\n    GGML_UNUSED(batch_size);',
        src,
    )
    if "int batch_size" not in src:
        print(f"ERROR: batch_size patch failed — signature not found in {path}", file=sys.stderr)
        sys.exit(1)
    print(f"  [OK] batch_size parameter")

if src == original:
    print(f"No patches needed for {path} — APIs have converged")
    sys.exit(0)

with open(path, "w") as f:
    f.write(src)

print(f"Patched {path} successfully")
