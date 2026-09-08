#!/usr/bin/env python3
"""Emit weak stubs for a whole API.

    python3 codegen/emit_runtime.py --api codegen/runtime_api.json
    python3 codegen/emit_runtime.py --api codegen/cublas_api.json \
        --out client/generated/cublas_stubs.cpp --header cublas_v2.h \
        --result-type cublasStatus_t --error CUBLAS_STATUS_NOT_SUPPORTED

The runtime shim is a translation layer, not a forwarding one: each function
that matters is hand-written in client/cudart_impl.cpp in terms of driver API
calls, which our libcuda.so.1 then remotes. Translation is semantic, so there
is nothing to generate for those.

What generation is good for here is completeness. Emitting a weak, logging
definition of every runtime entry point means the shim exports the same symbol
set as the real libcudart, so nothing fails to load, and any function we have
not translated announces itself by name instead of being a missing symbol.
Hand-written definitions are strong and override these.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from emit import HEADER, Unmarshalable, decl_params  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="codegen/runtime_api.json")
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="client/generated/cudart_stubs.cpp")
    ap.add_argument("--header", default="cuda_runtime_api.h")
    ap.add_argument("--result-type", default="cudaError_t")
    ap.add_argument("--error", default="cudaErrorNotSupported")
    a = ap.parse_args()

    with open(a.api) as f:
        doc = json.load(f)

    lines = [
        HEADER,
        "// Weak definitions of every entry point in this API. Hand-written",
        "// implementations are strong and override them; these catch",
        "// everything else by name instead of failing to link.",
        "#include <%s>" % a.header,
        "",
        "#include \"client/rpc.h\"",
        "",
    ]

    emitted, skipped = 0, []
    for f in doc["functions"]:
        name = f["name"]
        try:
            params = decl_params(f)
        except Unmarshalable as e:
            skipped.append((name, str(e)))
            continue
        lines += [
            "extern \"C\" __attribute__((weak)) %s %s(%s) {"
            % (a.result_type, name, params),
            "  rgpu::unimplemented_rt(\"%s\");" % name,
            "  return %s;" % a.error,
            "}",
            "",
        ]
        emitted += 1

    out = os.path.join(a.root, a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print("wrote %s: %d stubs, %d undeclarable"
          % (a.out, emitted, len(skipped)), file=sys.stderr)
    for name, why in skipped[:10]:
        print("  skipped %s (%s)" % (name, why), file=sys.stderr)


if __name__ == "__main__":
    main()
