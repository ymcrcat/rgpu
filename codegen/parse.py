#!/usr/bin/env python3
"""Parse cuda.h into a JSON model of the driver API.

Runs inside the CUDA devel container, which supplies both cuda.h and libclang.
Emits one record per driver function with each parameter classified, so emit.py
can decide what it can generate and what needs a hand-written body.

    python3 codegen/parse.py --header /usr/local/cuda/include/cuda.h -o api.json
"""

import argparse
import glob
import json
import re
import sys

import clang.cindex as ci

# Parameter kinds. The classifier assigns these from the C type alone; anything
# it cannot decide becomes "unknown" and either gets an annotation or a stub.
SCALAR = "scalar"  # passed by value
HANDLE = "handle"  # opaque pointer, passed through as an integer
OUT_SCALAR = "out_scalar"  # T* the callee fills, T a scalar
OUT_HANDLE = "out_handle"  # H* the callee fills, H opaque
STRING = "string"  # const char*
UNKNOWN = "unknown"  # needs an annotation, else the function is stubbed

# Types that look like scalars but must never be treated as buffers.
_SCALAR_KINDS = {
    ci.TypeKind.BOOL,
    ci.TypeKind.CHAR_U, ci.TypeKind.UCHAR, ci.TypeKind.USHORT,
    ci.TypeKind.UINT, ci.TypeKind.ULONG, ci.TypeKind.ULONGLONG,
    ci.TypeKind.CHAR_S, ci.TypeKind.SCHAR, ci.TypeKind.SHORT,
    ci.TypeKind.INT, ci.TypeKind.LONG, ci.TypeKind.LONGLONG,
    ci.TypeKind.FLOAT, ci.TypeKind.DOUBLE,
    ci.TypeKind.ENUM,
}


def _is_scalar(t):
    return t.get_canonical().kind in _SCALAR_KINDS


def _is_opaque_record(t):
    """True for a struct we only ever see as an incomplete type: a handle."""
    c = t.get_canonical()
    if c.kind != ci.TypeKind.RECORD:
        return False
    decl = c.get_declaration()
    # An opaque CUDA handle is declared but never defined in the header.
    return not decl.is_definition()


def _sugar_pointee(t):
    """Pointee spelling preserving typedefs, so generated locals read as the
    header does: CUcontext rather than struct CUctx_st *."""
    if t.kind == ci.TypeKind.POINTER:
        p = t.get_pointee()
    else:
        p = t.get_canonical().get_pointee()
    # Drop a leading const so the spelling is usable as a local declaration.
    s = p.spelling
    return s[len("const "):] if s.startswith("const ") else s


def classify(t):
    """Classify a parameter type. Returns (kind, detail, pointee_spelling)."""
    canon = t.get_canonical()

    if _is_scalar(t):
        return SCALAR, None, None

    if canon.kind != ci.TypeKind.POINTER:
        # Structs by value, function pointers, arrays: not auto-generatable.
        return UNKNOWN, canon.kind.name, None

    pointee = canon.get_pointee()
    is_const = pointee.is_const_qualified()
    pk = pointee.get_canonical()

    # const char* is a name or a string option.
    if pk.kind in (ci.TypeKind.CHAR_S, ci.TypeKind.CHAR_U) and is_const:
        return STRING, None, None

    # An opaque handle: pointer to an incomplete struct. The whole type, e.g.
    # CUcontext, is the handle; it travels as an integer.
    if _is_opaque_record(pointee):
        return HANDLE, pointee.spelling, None

    # Pointer to a handle: an output handle, e.g. CUcontext*.
    if pk.kind == ci.TypeKind.POINTER:
        inner = pk.get_pointee()
        if _is_opaque_record(inner) and not is_const:
            return OUT_HANDLE, inner.spelling, _sugar_pointee(t)
        return UNKNOWN, "pointer-to-pointer", None

    # Pointer to a scalar: an output slot if writable, otherwise a buffer whose
    # length we cannot infer, so it needs an annotation.
    if _is_scalar(pointee):
        if is_const:
            return UNKNOWN, "const-scalar-buffer", _sugar_pointee(t)
        return OUT_SCALAR, pointee.spelling, _sugar_pointee(t)

    # void*, or a pointer to a defined struct. Both need annotations.
    if pk.kind == ci.TypeKind.VOID:
        return UNKNOWN, "void-pointer", None
    return UNKNOWN, "struct-pointer:" + pointee.spelling, _sugar_pointee(t)


def parse_aliases(header):
    """Collect `#define cuFoo cuFoo_v2` style aliases.

    cudart asks cuGetProcAddress for the base name plus a CUDA version and
    expects the versioned implementation back, so the lookup table has to know
    that cuMemAlloc means cuMemAlloc_v2. Reading the defines is more reliable
    than guessing by stripping suffixes.
    """
    aliases = {}
    # The stream-ordered entry points are wrapped:
    #   #define cuMemcpyHtoD  __CUDA_API_PTDS(cuMemcpyHtoD_v2)
    # Both wrappers expand to their argument in a normal build, so unwrap them.
    # Missing these is not cosmetic: cuBLAS and cudart ask cuGetProcAddress for
    # the base names, and an unanswered request stops them initialising.
    pat = re.compile(
        r"^\s*#define\s+(cu[A-Za-z0-9_]+)\s+"
        r"(?:__CUDA_API_PT(?:DS|SZ)\(\s*)?"
        r"(cu[A-Za-z0-9_]+)"
        r"\s*\)?\s*$")
    with open(header, "r", errors="replace") as f:
        for line in f:
            m = pat.match(line)
            if m and m.group(1) != m.group(2):
                aliases[m.group(1)] = m.group(2)
    return aliases


def builtin_include_dirs():
    """clang's own headers (stddef.h and friends). The pip libclang wheel does
    not ship them, so find whatever the system has."""
    dirs = []
    for pat in ("/usr/lib/llvm-*/lib/clang/*/include",
                "/usr/lib/clang/*/include"):
        dirs.extend(sorted(glob.glob(pat)))
    return dirs


def parse(header, extra_args, prefix="cu", result_type="CUresult"):
    index = ci.Index.create()
    args = ["-x", "c", "-std=c11"]
    args += ["-I" + d for d in builtin_include_dirs()]
    args += list(extra_args)
    tu = index.parse(header, args=args)

    fatal = [d for d in tu.diagnostics if d.severity >= ci.Diagnostic.Error]
    if fatal:
        for d in fatal[:10]:
            print("clang: %s" % d.spelling, file=sys.stderr)
        raise SystemExit("cuda.h did not parse cleanly")

    funcs = []
    seen = set()
    for cur in tu.cursor.get_children():
        if cur.kind != ci.CursorKind.FUNCTION_DECL:
            continue
        name = cur.spelling
        if not name.startswith(prefix) or name in seen:
            continue
        # An API is the set of functions returning its error type: CUresult for
        # the driver, cudaError_t for the runtime.
        if cur.result_type.spelling != result_type:
            continue
        # Some headers define small helpers inline. Those already have a body,
        # so emitting our own would be a redefinition. cublasLt.h does this for
        # its *Init helpers.
        if cur.is_definition():
            continue
        seen.add(name)

        params = []
        for i, p in enumerate(cur.get_arguments()):
            kind, detail, pointee = classify(p.type)
            params.append({
                # Some declarations omit parameter names; synthesize one so
                # generated code always has something to refer to.
                "name": p.spelling or ("arg%d" % i),
                "type": p.type.spelling,
                "kind": kind,
                "detail": detail,
                "pointee": pointee,
            })
        funcs.append({
            "name": name,
            "params": params,
            # True when every parameter classified without help.
            "auto": all(p["kind"] != UNKNOWN for p in params),
        })

    funcs.sort(key=lambda f: f["name"])
    return funcs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--header", default="/usr/local/cuda/include/cuda.h")
    ap.add_argument("--libclang", default=None, help="path to libclang.so")
    ap.add_argument("-I", dest="includes", action="append", default=[])
    ap.add_argument("-o", dest="out", default="-")
    ap.add_argument("--prefix", default="cu",
                    help="function name prefix: cu for the driver API, "
                         "cuda for the runtime API")
    ap.add_argument("--result-type", default="CUresult",
                    help="CUresult for the driver API, cudaError_t for the "
                         "runtime API")
    a = ap.parse_args()

    if a.libclang:
        ci.Config.set_library_file(a.libclang)

    funcs = parse(a.header, ["-I" + i for i in a.includes],
                  prefix=a.prefix, result_type=a.result_type)
    aliases = parse_aliases(a.header)
    doc = {"header": a.header, "prefix": a.prefix,
           "result_type": a.result_type,
           "functions": funcs, "aliases": aliases}
    text = json.dumps(doc, indent=1)
    if a.out == "-":
        print(text)
    else:
        with open(a.out, "w") as f:
            f.write(text)
    auto = sum(1 for f in funcs if f["auto"])
    print("parsed %d driver functions, %d fully auto-classified"
          % (len(funcs), auto), file=sys.stderr)


if __name__ == "__main__":
    main()
