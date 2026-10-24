"""WebSocket server streaming CaptionLineFeed messages to Godot clients."""
from __future__ import annotations

import json

from websockets.asyncio.server import broadcast, serve

from caption_feed import DEFAULT_PORT



class CaptionServer:
    """Async context manager. On connect a client gets the feed's snapshot; afterwards every
    message the feed emits is broadcast. a connecting client also gets the signs still in flight;
    decoded client messages go to the feed, which ignores all but valid acks."""

    def __init__(self, feed, host: str = "0.0.0.0", port: int = DEFAULT_PORT):
        self.feed, self.host, self._port = feed, host, port
        self._server = None

    @property
    def port(self) -> int:
        return self._server.sockets[0].getsockname()[1]

    async def __aenter__(self):
        self._server = await serve(self._handle, self.host, self._port)
        self.feed.emit = self._broadcast
        return self

    async def __aexit__(self, *exc):
        self.feed.emit = lambda msg: None
        self._server.close()
        await self._server.wait_closed()

    def _broadcast(self, msg: dict):
        broadcast(self._server.connections, json.dumps(msg, ensure_ascii=False))

    async def _handle(self, ws):
        snapshot, playing = self.feed.snapshot(), self.feed.in_flight_signs()
        await ws.send(json.dumps(snapshot, ensure_ascii=False))
        for sign in playing:  # signs sent before this client connected
            await ws.send(json.dumps(sign, ensure_ascii=False))
        async for raw in ws:
            try:
                self.feed.client_message(json.loads(raw))
            except ValueError:
                continue
