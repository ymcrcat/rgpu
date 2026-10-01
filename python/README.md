# rgpu

A PyTorch device whose tensors live on a remote GPU. Train on a laptop with
`device="rgpu"` while the work runs on a GPU host you reach over SSH.

```sh
pip install rgpu
```

The GPU host runs `rgpu-opserver`; the laptop runs your script through
`rgpu-run --server 127.0.0.1:9720 python train.py`. The protocol has no
authentication, so keep the server on localhost and reach it through an SSH
tunnel.

Documentation: https://rgpu.pages.dev/docs/
