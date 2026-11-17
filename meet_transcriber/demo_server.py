#!/usr/bin/env python3
"""Run the real Godot caption server with a scripted conversation instead of Google Meet.

    python demo_server.py [--port 8765] [--no-broadcast] [--once]

Uses the same pieces as `meet_transcriber.py --godot` (caption feed, ASL gloss translation,
sign library, sign pacing, LAN discovery), so the Godot app sees exactly what a live meeting
sends: snapshot, floor changes, interim and final lines, side comments and paced signs.
Client acks are printed so you can check the app acknowledges every sign.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys

from caption_feed import DEFAULT_PORT, CaptionLineFeed
from caption_server import CaptionServer
from meet_transcriber import _sign_sources, announce_lan

# (source id, speaker, text). Ann speaks first, so she holds the floor and is signed;
# Bob's line while she holds it is a side comment (text only).
SCRIPT = [
    (1, "Ann", "Hello, can you help my family?"),
    (2, "Bob", "sure, one second"),
    (1, "Ann", "I want a drink because it is cold"),
    (1, "Ann", "Who is your brother?"),
]
WORD_DELAY = 0.25   # seconds between words of an interim caption
LINE_PAUSE = 1.0    # pause after a line is final
LOOP_PAUSE = 6.0    # pause before the script starts again


async def speak(feed: CaptionLineFeed, source: int, speaker: str, text: str):
    words = text.split()
    for n in range(1, len(words) + 1):
        feed.interim(source, speaker, " ".join(words[:n]))
        await asyncio.sleep(WORD_DELAY)
    feed.final(source, speaker, text)
    await asyncio.sleep(LINE_PAUSE)


async def tick_forever(feed: CaptionLineFeed):
    while True:
        feed.tick()
        await asyncio.sleep(0.1)


async def main(args) -> int:
    print("Loading the ASL translator and sign library (first run is slow)...", flush=True)
    translate, resolve = _sign_sources()
    feed = CaptionLineFeed(lambda msg: None, translate, resolve=resolve)

    original = feed.client_message

    def log_ack(msg):
        print("client ->", msg, flush=True)
        original(msg)

    feed.client_message = log_ack

    async with contextlib.AsyncExitStack() as stack:
        server = await stack.enter_async_context(CaptionServer(feed, port=args.port))
        await announce_lan(stack, server.port, broadcast=not args.no_broadcast)
        ticker = asyncio.create_task(tick_forever(feed))
        try:
            while True:
                for source, speaker, text in SCRIPT:
                    print(f"{speaker}: {text}", flush=True)
                    await speak(feed, source, speaker, text)
                if args.once:
                    while not feed._player.idle():
                        await asyncio.sleep(0.2)
                    await asyncio.sleep(2)
                    return 0
                await asyncio.sleep(LOOP_PAUSE)
        finally:
            ticker.cancel()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--no-broadcast", action="store_true", help="don't announce on the LAN (UDP 8766)")
    ap.add_argument("--once", action="store_true", help="play the script once, then exit")
    try:
        sys.exit(asyncio.run(main(ap.parse_args())))
    except KeyboardInterrupt:
        pass
