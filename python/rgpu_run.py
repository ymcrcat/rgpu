"""rgpu-run: run a command against a remote GPU, without editing the command.

    rgpu-run --host user@gpuhost python train.py
    rgpu-run --server 127.0.0.1:9720 python train.py   # tunnel already open

With --host it opens an ssh tunnel to the server, points RGPU_OPSERVER at the
local end, runs the command, and closes the tunnel afterwards. The script
itself still has to `import rgpu` and ask for `device="rgpu"`: this moves the
plumbing, not the device.

The tunnel is what makes this safe to use. rgpu-opserver has no authentication
of any kind, so it binds 127.0.0.1 on the GPU host and the forward is the only
way in. The local end is bound to 127.0.0.1 too, so nothing else on your
network can reach it either.

The server needs its own PyTorch, of the same major.minor as yours. That is a
real prerequisite and this cannot hide it; scripts/opserver_pod.sh sets one up
on a fresh box.
"""

import argparse
import os
import socket
import subprocess
import sys
import time

DEFAULT_PORT = 9720


def free_port():
    """A port nothing is listening on, for the local end of the tunnel."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def tunnel_command(host, ssh_port, identity, local_port, remote_port):
    """The ssh that forwards local_port to remote_port on the GPU host.

    Both ends are 127.0.0.1 on purpose. The remote end because that is where
    rgpu-opserver listens, and the local end because an unauthenticated GPU
    reachable from the rest of the network is the thing to avoid.
    """
    argv = ["ssh", "-N", "-o", "ExitOnForwardFailure=yes"]
    if identity:
        argv += ["-i", identity]
    if ssh_port:
        argv += ["-p", str(ssh_port)]
    argv += ["-L", f"{local_port}:127.0.0.1:{remote_port}", host]
    return argv


def wait_for(port, deadline):
    """True once something accepts on the local end of the tunnel."""
    while time.monotonic() < deadline:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.3).close()
            return True
        except OSError:
            time.sleep(0.1)
    return False


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="rgpu-run", description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument("--host", help="ssh destination of the GPU host, e.g. user@gpuhost")
    where.add_argument("--server", metavar="HOST:PORT",
                       help="an already-reachable server; no tunnel is opened")
    parser.add_argument("--ssh-port", type=int, help="ssh port on the GPU host")
    parser.add_argument("-i", "--identity", help="ssh private key")
    parser.add_argument("--remote-port", type=int, default=DEFAULT_PORT,
                        help=f"port rgpu-opserver listens on, default {DEFAULT_PORT}")
    parser.add_argument("--timeout", type=float, default=30,
                        help="seconds to wait for the tunnel, default 30")
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="the command to run")
    args = parser.parse_args(argv)

    command = [a for a in args.command if a != "--"]
    if not command:
        parser.error("give a command to run, e.g. rgpu-run --host h python train.py")

    if args.server:
        return spawn(command, args.server)

    local = free_port()
    tunnel = subprocess.Popen(
        tunnel_command(args.host, args.ssh_port, args.identity, local, args.remote_port))
    try:
        if not wait_for(local, time.monotonic() + args.timeout):
            tunnel.terminate()
            # ssh has already printed why on its own stderr; saying it could
            # not connect a second time would just bury that.
            print(f"rgpu-run: no tunnel to {args.host} within {args.timeout:g}s",
                  file=sys.stderr)
            return 1
        return spawn(command, f"127.0.0.1:{local}")
    finally:
        tunnel.terminate()
        try:
            tunnel.wait(timeout=5)
        except subprocess.TimeoutExpired:
            tunnel.kill()


def spawn(command, address):
    """Run the command with RGPU_OPSERVER pointed at `address`.

    A child rather than an exec, so the tunnel is still ours to close when the
    command is done.
    """
    env = {**os.environ, "RGPU_OPSERVER": address}
    try:
        return subprocess.call(command, env=env)
    except FileNotFoundError:
        print(f"rgpu-run: no such command: {command[0]}", file=sys.stderr)
        return 127
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
