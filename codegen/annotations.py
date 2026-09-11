"""Per-function facts the C header cannot express.

A reduced form of AvA's LAPIS vocabulary, as a plain dict rather than a DSL with
its own compiler, because we target exactly one API.

Parameter annotations
---------------------
    in_buffer(expr)   caller's bytes, length given by `expr` over other params
    out_buffer(expr)  callee fills these bytes, length given by `expr`
    pod_in / pod_out  fixed-size struct with no pointers inside; copy verbatim
    fatbin            image whose length is parsed from its own header
    ignore            never forwarded (e.g. a reserved parameter)

Execution class, following LAPIS
--------------------------------
    "sync"   returns data or an immediately observable status. Round trip.
    "async"  effect only observable at a later synchronization point. May be
             sent fire-and-forget once batching is enabled.
    "flush"  returns immediately but must first submit everything queued.

`sync_stream` names a stream parameter the server must synchronize before it
serializes any output buffer. An asynchronous device-to-host copy only enqueues
work, so without this the server would read the destination buffer before the
copy had run and send back whatever happened to be there.

`record` marks a call that establishes durable server-side state. Replaying the
recorded set reconstructs a session, which is what mrCUDA-style migration and
reconnect-after-drop would need. Nothing consumes it yet; tagging is cheap and
retrofitting the tags later would not be.
"""

# Functions we write by hand. Generated code will not define these.
HANDWRITTEN = {
    # Resolves driver entry points; must return our pointers, not the real ones.
    # This is the reason exported symbols alone are not enough on CUDA 11.3+.
    "cuGetProcAddress",
    "cuGetProcAddress_v2",
    # Return pointers to static strings owned by the driver.
    "cuGetErrorString",
    "cuGetErrorName",
    # Argument marshalling depends on a per-function parameter layout the
    # server has to look up. See docs/.../design.md, hard problem 2.
    "cuLaunchKernel",
    "cuLaunchCooperativeKernel",
    # Host allocations live on the client, not the server.
    "cuMemAllocHost_v2",
    "cuMemHostAlloc",
    "cuMemFreeHost",
    "cuMemHostRegister_v2",
    "cuMemHostUnregister",
    # Version negotiation is answered locally.
    "cuDriverGetVersion",
    # One output is a pointer to an array the driver owns, which cannot cross
    # a wire as a pointer. The server copies the array out and the client keeps
    # it for as long as the driver would have.
    "cuStreamGetCaptureInfo_v2",
    # Its count is the caller's capacity on the way in and the number written
    # on the way out, which the generator cannot express.
    "cuGraphGetNodes",
    # Its parameter is read as well as written, which the generator cannot
    # see: it sent no value, so the mode asked for was always zero.
    "cuThreadExchangeStreamCaptureMode",
    # Undocumented table of internal driver function pointers. cudart asks for
    # it immediately after cuInit. Hand-written so we can log which table is
    # wanted and control exactly what we answer.
    "cuGetExportTable",
}

# Hand-written on the client only. The server side is still generated, because
# what these do differently is answer locally where they can; when they do go
# to the server they go as the ordinary call.
# The primary context's state is answered from a client-side cache while we
# hold a reference to it. PyTorch asks for it hundreds of times per inference,
# and it was 45% of all round trips before this. Retain, release, reset and
# set-flags come along because they are what changes the answer.
HANDWRITTEN_CLIENT = {
    # Triton asks for every kernel argument's device pointer on every launch:
    # 144 round trips per compiled inference. We made the allocations, so the
    # answer is already here. Alloc and free come along to keep the table.
    "cuPointerGetAttribute",
    "cuMemAlloc_v2",
    "cuMemFree_v2",
    "cuDevicePrimaryCtxGetState",
    "cuDevicePrimaryCtxRetain",
    "cuDevicePrimaryCtxRelease_v2",
    "cuDevicePrimaryCtxReset_v2",
    "cuDevicePrimaryCtxSetFlags_v2",
}

# Explicitly refused, with the reason surfaced in the log. These cannot work
# across a network and failing loudly beats corrupting silently.
UNSUPPORTED = {
    # CUlaunchConfig carries an attribute array we do not marshal yet. Used for
    # thread-block clusters on Hopper and later; nothing in milestone 1 needs it.
    "cuLaunchKernelEx": "launch config attributes not marshalled yet",
    "cuMemAllocManaged": "managed memory cannot span a network",
    "cuMemHostGetDevicePointer_v2": "zero-copy host mapping cannot span a network",
    "cuMemHostGetDevicePointer": "zero-copy host mapping cannot span a network",
    "cuIpcOpenMemHandle_v2": "IPC handles are host-local",
    "cuIpcGetMemHandle": "IPC handles are host-local",
    "cuIpcOpenEventHandle": "IPC handles are host-local",
    "cuIpcGetEventHandle": "IPC handles are host-local",
}

# name -> {"params": {param: annotation}, "exec": class, "record": bool}
ANNOTATIONS = {
    # ---- device queries -------------------------------------------------
    "cuDeviceGetName": {"params": {"name": "out_buffer(len)"}},
    "cuDeviceGetUuid": {"params": {"uuid": "pod_out"}},
    "cuDeviceGetUuid_v2": {"params": {"uuid": "pod_out"}},
    "cuDeviceGetLuid": {"params": {"luid": "out_buffer(8)",
                                   "deviceNodeMask": "out_scalar"}},

    # ---- context --------------------------------------------------------
    "cuCtxCreate_v2": {"record": True},
    "cuDevicePrimaryCtxRetain": {"record": True},
    "cuCtxSynchronize": {"exec": "flush"},

    # ---- memory ---------------------------------------------------------
    "cuMemAlloc_v2": {"record": True},
    "cuMemcpyHtoD_v2": {"params": {"srcHost": "in_buffer(ByteCount)"}},
    "cuMemcpyDtoH_v2": {"params": {"dstHost": "out_buffer(ByteCount)"}},
    "cuMemcpyHtoDAsync_v2": {"params": {"srcHost": "in_buffer(ByteCount)"},
                             "exec": "async"},
    # Not async in our sense. The caller may read dstHost after any later
    # synchronization, so the bytes have to travel with the reply, which means
    # the server has to wait for the copy it just enqueued.
    "cuMemcpyDtoHAsync_v2": {"params": {"dstHost": "out_buffer(ByteCount)"},
                             "sync_stream": "hStream"},

    # The value size depends on which attribute is asked for, so the size
    # expression calls a helper rather than naming another parameter.
    "cuPointerGetAttribute": {
        "params": {"data": "out_buffer(rgpu::pointer_attr_size(attribute))"},
    },

    # ---- modules and kernels --------------------------------------------
    "cuModuleLoadData": {"params": {"image": "fatbin"}, "record": True},
    "cuModuleLoadFatBinary": {"params": {"fatCubin": "fatbin"}, "record": True},
    "cuLibraryLoadData": {
        "params": {
            "code": "fatbin",
            # Option arrays are unused by cudart in practice; refuse non-empty.
            "jitOptions": "ignore", "jitOptionsValues": "ignore",
            "libraryOptions": "ignore", "libraryOptionValues": "ignore",
        },
        "record": True,
    },
    "cuModuleGetFunction": {"record": True},
    "cuLibraryGetKernel": {"record": True},

    # ---- streams and events ---------------------------------------------
    "cuStreamCreate": {"record": True},
    "cuStreamSynchronize": {"exec": "flush"},
    "cuStreamWaitEvent": {"exec": "async"},
    "cuEventRecord": {"exec": "async"},
    "cuEventSynchronize": {"exec": "flush"},
    "cuEventCreate": {"record": True},
}

# Memsets are all fire-and-forget on a stream.
for _f in ("cuMemsetD8Async", "cuMemsetD16Async", "cuMemsetD32Async",
           "cuMemsetD2D8Async", "cuMemsetD2D16Async", "cuMemsetD2D32Async"):
    ANNOTATIONS.setdefault(_f, {})["exec"] = "async"


def for_function(name):
    """Annotation record for `name`, with defaults filled in."""
    a = dict(ANNOTATIONS.get(name, {}))
    a.setdefault("params", {})
    a.setdefault("exec", "sync")
    a.setdefault("record", False)
    a.setdefault("sync_stream", None)
    return a
