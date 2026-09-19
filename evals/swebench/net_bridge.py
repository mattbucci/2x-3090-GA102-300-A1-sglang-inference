#!/usr/bin/env python3
"""net_bridge.py -- the two halves of the rollout network bridge.

Rollout containers run with `--network none` (answer-leakage fix, 2026-09-19:
with `--network=host` the scaffolds' webfetch / curl reached GitHub and 21 %
of qwen38 opencode instances downloaded their own PR diff). The container
keeps its loopback, so the scaffold's configured http://127.0.0.1:<port>
still works through this bridge while every other destination fails at
connect time (no routes, no DNS):

    host side (docker_rollout.py, lives for the run):
        python3 net_bridge.py host <unix socket> <server host> <server port>
    container side (started by the inner script before the scaffold):
        python3 net_bridge.py container <port> <unix socket>

The host half listens on a unix socket that docker bind-mounts into the
container and connects each client to the SGLang TCP port; the container half
listens on 127.0.0.1:<port> and connects each client to the mounted socket.
stdlib only (no socat on either side); half-closes are propagated with
write_eof so streaming responses are not cut short by a client that finished
sending. Runs under the images' base python (/opt/miniconda3/bin/python3),
not the per-instance testbed env, which can be as old as 3.5.
"""
import asyncio
import os
import stat
import sys


async def _pump(src, dst):
    try:
        while True:
            data = await src.read(65536)
            if not data:
                break
            dst.write(data)
            await dst.drain()
    except Exception:
        pass
    finally:
        try:
            if dst.can_write_eof():
                dst.write_eof()
            else:
                dst.close()
        except Exception:
            pass


async def _splice(reader, writer, opener):
    try:
        ureader, uwriter = await opener()
    except Exception as e:
        print("net_bridge: upstream connect failed: %s" % e, file=sys.stderr, flush=True)
        writer.close()
        return
    await asyncio.gather(_pump(reader, uwriter), _pump(ureader, writer))
    for w in (writer, uwriter):
        try:
            w.close()
        except Exception:
            pass


def main():
    mode = sys.argv[1]
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    if mode == "host":
        sock, host, port = sys.argv[2], sys.argv[3], int(sys.argv[4])
        try:
            os.unlink(sock)
        except FileNotFoundError:
            pass
        server = loop.run_until_complete(asyncio.start_unix_server(
            lambda r, w: _splice(r, w, lambda: asyncio.open_connection(host, port)), sock))
        # the container's root must be able to connect; the socket lives in a
        # user-private runtime dir so the mode is not a wider exposure
        os.chmod(sock, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH)
    elif mode == "container":
        port, sock = int(sys.argv[2]), sys.argv[3]
        server = loop.run_until_complete(asyncio.start_server(
            lambda r, w: _splice(r, w, lambda: asyncio.open_unix_connection(sock)), "127.0.0.1", port))
    else:
        sys.exit("usage: net_bridge.py host <sock> <host> <port> | container <port> <sock>")
    try:
        loop.run_forever()
    finally:
        server.close()


if __name__ == "__main__":
    main()
