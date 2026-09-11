import os
import socket
import subprocess
import sys
import time

import pytest

import rgpu  # noqa: F401  registers the "rgpu" device for every test module


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_server(port=None, env=None, device="cpu"):
    port = port or free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "rgpu.server", "--device", device, "--port", str(port)],
        env={**os.environ, **(env or {})})
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            return proc, f"127.0.0.1:{port}"
        except OSError:
            if proc.poll() is not None:
                raise RuntimeError("rgpu-opserver exited while starting")
            time.sleep(0.1)
    proc.kill()
    raise RuntimeError("rgpu-opserver did not start within 30 s")


@pytest.fixture(scope="session", autouse=True)
def opserver():
    """A CPU server for the whole run, unless RGPU_OPSERVER names a real one."""
    if os.environ.get("RGPU_OPSERVER"):
        yield None
        return
    proc, address = start_server(device=os.environ.get("RGPU_TEST_DEVICE", "cpu"))
    os.environ["RGPU_OPSERVER"] = address
    yield proc
    proc.terminate()
    proc.wait(timeout=10)
