"""Connect stock JAX on a Mac to a remote GPU through the IFRT proxy.

    # on the GPU host
    rgpu_ifrt_server --port=12345 --backend=gpu
    # on the Mac
    ssh -N -L 12345:127.0.0.1:12345 user@gpuhost
    IFRT_PROXY_USE_INSECURE_GRPC_CREDENTIALS=true python client_example.py

No plugin, no shim, no code generation: jaxlib already carries the proxy
client, and this points it at a server.

Three things will each waste an afternoon if you get them wrong.

1. **Import jax before touching the proxy.** Calling get_client() without
   having imported jax segfaults the interpreter - exit 139, no traceback, no
   message. The jax runtime has to be initialised first.

2. **IFRT_PROXY_USE_INSECURE_GRPC_CREDENTIALS must be exactly "true".** It is
   needed on both client and server. The client otherwise uses ALTS and every
   connection is refused with "Invalid credentials"; "1", "TRUE" and "yes" are
   silently ignored, which looks identical to not having set it at all.

3. **The server's XLA revision must match the client's jaxlib.** The proxy
   performs a version handshake. See build_on_host.sh for the pin.
"""

import os
import sys

import numpy as np

import jax                      # first, and not optional: see above
import jax.numpy as jnp
from jax._src import xla_bridge as xb
from jax.extend.backend import ifrt_proxy

ADDRESS = os.environ.get("RGPU_IFRT", "grpc://127.0.0.1:12345")
NAME = "rgpu_proxy"


def connect(address=ADDRESS, name=NAME, priority=500):
    """Register the remote server as a JAX backend and return it."""
    if not os.environ.get("IFRT_PROXY_USE_INSECURE_GRPC_CREDENTIALS") == "true":
        print("warning: IFRT_PROXY_USE_INSECURE_GRPC_CREDENTIALS is not 'true'; "
              "the connection will be refused as 'Invalid credentials'",
              file=sys.stderr)

    def factory():
        options = ifrt_proxy.ClientConnectionOptions()
        options.connection_timeout_in_seconds = 20
        return ifrt_proxy.get_client(address, options)

    xb.register_backend_factory(name, factory, priority=priority)
    return xb.get_backend(name)


def main():
    backend = connect()
    print("connected to", ADDRESS)
    print("  devices:", backend.devices())

    device = backend.devices()[0]
    host = np.arange(12, dtype=np.float32).reshape(3, 4)

    x = jax.device_put(host, device)
    print("  placed  :", x.shape, "on", x.device)

    y = jax.jit(lambda a: (a @ a.T) * 2.0)(x)
    print("  computed:", np.asarray(y).tolist())

    want = (host @ host.T) * 2.0
    ok = np.allclose(np.asarray(y), want)
    print("  matches a local numpy reference:", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
