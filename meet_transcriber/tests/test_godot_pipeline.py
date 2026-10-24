"""Tracker -> GodotSink -> feed -> server -> real WebSocket client."""
import asyncio
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from caption_feed import CaptionLineFeed  # noqa: E402
from caption_server import CaptionServer  # noqa: E402
from meet_transcriber import CaptionTracker, GodotSink  # noqa: E402
from websockets.asyncio.client import connect  # noqa: E402


class QuietSink:
    def interim(self, *a): pass
    def final(self, *a): pass
    def close(self): pass


def snap(text, bid=1, speaker="Ann"):
    return [{"id": bid, "speaker": speaker, "text": text}]


def run(coro):
    return asyncio.run(asyncio.wait_for(coro, 5))


def summary(msgs):
    return [(m["id"], m["speaker"], m["text"], m["final"]) for m in msgs if m["type"] == "line"]


class PipelineTests(unittest.TestCase):
    def pipeline(self):
        feed = CaptionLineFeed(lambda m: None)
        tracker = CaptionTracker(GodotSink(QuietSink(), feed, echo_interim=False),
                                 idle=0.8, settle=0.3, live=True)
        return feed, tracker

    @staticmethod
    async def take(ws, n):
        """The next n line messages (floor changes are skipped)."""
        msgs = []
        while len(msgs) < n:
            msg = json.loads(await ws.recv())
            if msg["type"] == "line":
                msgs.append(msg)
        return msgs

    def test_client_receives_lines_driven_by_the_tracker(self):
        async def scenario():
            feed, t = self.pipeline()
            async with CaptionServer(feed, host="127.0.0.1", port=0) as server:
                async with connect(f"ws://127.0.0.1:{server.port}") as ws:
                    await ws.recv()
                    t.update(snap("hello"), now=0)
                    t.update(snap("hello there"), now=0.2)
                    t.tick(1.5)
                    return await self.take(ws, 3)

        self.assertEqual(summary(run(scenario())), [
            (1, "Ann", "hello", False), (1, "Ann", "hello there", False), (1, "Ann", "hello there", True)])

    def test_reused_caption_source_after_silent_close_gets_a_new_line_id(self):
        async def scenario():
            feed, t = self.pipeline()
            async with CaptionServer(feed, host="127.0.0.1", port=0) as server:
                async with connect(f"ws://127.0.0.1:{server.port}") as ws:
                    await ws.recv()
                    t.update(snap("hello there"), now=0)
                    t.tick(1.0)                                  # line 1 final
                    t.update(snap("hello there friend"), now=1.1)  # line 2 open: "friend"
                    t.update(snap("hello there"), now=1.2)       # revision shrank: nothing new
                    t.update([], now=1.3)                        # block gone, no final from tracker
                    t.update(snap("brand new"), now=2.0)
                    return await self.take(ws, 5)

        self.assertEqual(summary(run(scenario())), [
            (1, "Ann", "hello there", False), (1, "Ann", "hello there", True),
            (2, "Ann", "friend", False), (2, "Ann", "friend", True),
            (3, "Ann", "brand new", False)])

    def test_speaker_change_after_silent_close_does_not_leak_into_old_line(self):
        async def scenario():
            feed, t = self.pipeline()
            async with CaptionServer(feed, host="127.0.0.1", port=0) as server:
                async with connect(f"ws://127.0.0.1:{server.port}") as ws:
                    await ws.recv()
                    t.update(snap("hello there"), now=0)
                    t.tick(1.0)
                    t.update(snap("hello there friend"), now=1.1)
                    t.update(snap("hello there"), now=1.2)
                    t.update(snap("yo", speaker="Bob"), now=1.3)
                    return (await self.take(ws, 5))[3:]

        self.assertEqual(summary(run(scenario())), [
            (2, "Ann", "friend", True), (3, "Bob", "yo", False)])

    def test_flush_before_server_close_reaches_the_client(self):
        async def scenario():
            feed, t = self.pipeline()
            server = CaptionServer(feed, host="127.0.0.1", port=0)
            async with server:
                ws = await connect(f"ws://127.0.0.1:{server.port}")
                await ws.recv()
                t.update(snap("last words"), now=0)
                t.flush_all()
            msgs = [json.loads(m) async for m in ws]
            await ws.close()
            return msgs

        msgs = run(scenario())
        self.assertEqual(summary(msgs)[-1], (1, "Ann", "last words", True))


if __name__ == "__main__":
    unittest.main()
