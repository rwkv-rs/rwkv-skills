#!/usr/bin/env python3
"""Forward a local TCP port to another host without restarting Docker."""

from __future__ import annotations

import argparse
import select
import socket
import socketserver


class _ForwardingHandler(socketserver.BaseRequestHandler):
    target: tuple[str, int]

    def handle(self) -> None:
        with socket.create_connection(self.target, timeout=15) as upstream:
            peers = (self.request, upstream)
            for peer in peers:
                peer.setblocking(False)
            while True:
                readable, _, _ = select.select(peers, (), (), 60)
                if not readable:
                    continue
                for source in readable:
                    data = source.recv(1024 * 1024)
                    if not data:
                        return
                    destination = upstream if source is self.request else self.request
                    destination.sendall(data)


class _ThreadingServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument("--target-host", required=True)
    parser.add_argument("--target-port", type=int, required=True)
    args = parser.parse_args()

    _ForwardingHandler.target = (args.target_host, args.target_port)
    with _ThreadingServer((args.listen_host, args.listen_port), _ForwardingHandler) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
