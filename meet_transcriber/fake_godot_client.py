#!/usr/bin/env python3
"""Demo stand-in for the Godot client: prints every message from the caption server.

    python fake_godot_client.py [ws://localhost:8765]
"""
import asyncio
import sys

from websockets.asyncio.client import connect


async def main(url: str):
    async with connect(url) as ws:
        async for msg in ws:
            print(msg, flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:8765"))
    except KeyboardInterrupt:
        pass
