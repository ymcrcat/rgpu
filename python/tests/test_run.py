"""rgpu-run: put a command on a remote GPU without editing it.

The tunnel needs a real ssh host, so what is tested here is everything
around it: the ssh command it would run, the address it hands the child,
and that a command really does compute on the server when pointed at one.
"""

import os
import subprocess
import sys

import pytest

import rgpu_run as run


def test_tunnel_command_carries_the_identity_and_the_port():
    argv = run.tunnel_command("user@gpuhost", ssh_port=2222,
                              identity="/k/id_ed25519",
                              local_port=9999, remote_port=9720)
    assert argv[0] == "ssh"
    assert "-N" in argv                      # no shell, just the forward
    assert "user@gpuhost" == argv[-1]
    assert "-L" in argv
    assert "9999:127.0.0.1:9720" in argv     # never 0.0.0.0: localhost only
    assert "-p" in argv and "2222" in argv
    assert "-i" in argv and "/k/id_ed25519" in argv


def test_tunnel_command_leaves_out_what_was_not_given():
    argv = run.tunnel_command("gpuhost", ssh_port=None, identity=None,
                              local_port=9999, remote_port=9720)
    assert "-i" not in argv
    assert "-p" not in argv
    assert "9999:127.0.0.1:9720" in argv


def test_a_free_port_is_actually_free():
    port = run.free_port()
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", port))          # raises if it was taken


def run_cli(*args, env=None):
    return subprocess.run([sys.executable, "-m", "rgpu_run", *args],
                          capture_output=True, text=True,
                          env={**os.environ, **(env or {})})


def test_no_command_is_a_usage_error():
    done = run_cli("--server", "127.0.0.1:9720")
    assert done.returncode == 2
    assert "usage" in (done.stderr + done.stdout).lower()


def test_server_mode_hands_the_address_to_the_child():
    """--server skips the tunnel: the address goes straight to the child, so a
    tunnel someone already has open is not worth a second one."""
    done = run_cli("--server", "10.0.0.5:9999", sys.executable, "-c",
                   "import os; print(os.environ['RGPU_OPSERVER'])")
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "10.0.0.5:9999"


def test_the_child_exit_status_is_passed_through():
    done = run_cli("--server", "10.0.0.5:9999", sys.executable, "-c",
                   "raise SystemExit(7)")
    assert done.returncode == 7


def test_a_real_tensor_op_runs_through_rgpu_run(opserver):
    """The whole point: an unmodified script, pointed at a server, computes."""
    done = run_cli("--server", os.environ["RGPU_OPSERVER"], sys.executable, "-c",
                   "import rgpu, torch;"
                   "x = torch.ones(4, 4, device='rgpu');"
                   "print(int((x @ x).sum().item()))")
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "64"
