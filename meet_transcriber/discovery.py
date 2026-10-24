"""UDP LAN announcement so the phone app finds the caption WebSocket without typing an address."""
from __future__ import annotations

import asyncio
import json
import socket

DISCOVERY_PORT = 8766
SERVICE = "silentvoice-captions"
PROTOCOL_VERSION = 1


def lan_ip() -> str:
    """The address other machines on the LAN reach this PC at (loopback if offline).
    Connecting a UDP socket sends nothing; it only makes the OS pick the outbound interface."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("192.0.2.1", 9))
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def ws_url(ip: str, port: int) -> str:
    return f"ws://{ip}:{port}"


def announcement(ip: str, port: int) -> dict:
    return {"v": PROTOCOL_VERSION, "service": SERVICE, "ws": ws_url(ip, port)}


class Announcer:
    """Async context manager broadcasting the announcement every `interval` seconds."""

    def __init__(self, ip: str, ws_port: int, target=("255.255.255.255", DISCOVERY_PORT),
                 interval: float = 1.0):
        self.payload = json.dumps(announcement(ip, ws_port)).encode()
        self.target, self.interval = target, interval
        self._sock = None
        self._task = None

    async def __aenter__(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self._sock.setblocking(False)
        self._task = asyncio.create_task(self._loop())
        return self

    async def __aexit__(self, *exc):
        self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)
        self._sock.close()

    async def _loop(self):
        loop = asyncio.get_running_loop()
        while True:
            try:
                await loop.sock_sendto(self._sock, self.payload, self.target)
            except OSError:  # no network right now; keep trying
                pass
            await asyncio.sleep(self.interval)
