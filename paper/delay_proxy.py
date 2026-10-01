"""A TCP proxy that delays each direction by a fixed time, for an RTT sweep.

    python delay_proxy.py --listen 9731 --target 127.0.0.1:9720 --one-way-ms 25

Every chunk is forwarded one_way_ms after it arrived, in order, so a request
and its reply together see 2 * one_way_ms of added round trip. Reading never
waits on the delay, so a stream is delayed and not throttled. Binds loopback
only: the rgpu protocol has no authentication.

ponytail: delay only. No jitter, loss or bandwidth limit; use tc netem for
those where the container allows it.
"""
import argparse
import asyncio
import socket


async def pump(reader, writer, delay):
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue()

    async def drain():
        while True:
            due, data = await queue.get()
            if data is None:
                break
            wait = due - loop.time()
            if wait > 0:
                await asyncio.sleep(wait)
            writer.write(data)
            await writer.drain()
        writer.close()

    task = asyncio.create_task(drain())
    try:
        while data := await reader.read(1 << 16):
            queue.put_nowait((loop.time() + delay, data))
    except ConnectionError:
        pass
    queue.put_nowait((0, None))
    await task


def nodelay(writer):
    writer.get_extra_info("socket").setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", type=int, required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--one-way-ms", type=float, required=True)
    args = ap.parse_args()
    host, port = args.target.rsplit(":", 1)
    delay = args.one_way_ms / 1e3

    async def serve(cr, cw):
        sr, sw = await asyncio.open_connection(host, int(port))
        nodelay(cw), nodelay(sw)
        await asyncio.gather(pump(cr, sw, delay), pump(sr, cw, delay))

    server = await asyncio.start_server(serve, "127.0.0.1", args.listen)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
