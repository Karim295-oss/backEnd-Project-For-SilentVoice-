import asyncio
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from caption_feed import ROLE_FLOOR, ROLE_SIDE, CaptionLineFeed  # noqa: E402
from caption_server import CaptionServer  # noqa: E402
from websockets.asyncio.client import connect  # noqa: E402


def run(coro):
    return asyncio.run(asyncio.wait_for(coro, 5))


class CaptionServerTests(unittest.TestCase):
    def test_client_gets_snapshot_then_live_lines(self):
        async def scenario():
            feed = CaptionLineFeed(lambda m: None)
            feed.final(1, "Ann", "earlier")
            async with CaptionServer(feed, host="127.0.0.1", port=0) as server:
                async with connect(f"ws://127.0.0.1:{server.port}") as ws:
                    snap = json.loads(await ws.recv())
                    feed.interim(2, "Bob", "hi there")
                    line = json.loads(await ws.recv())
            return snap, line

        snap, line = run(scenario())
        self.assertEqual(snap.pop("skeleton")["pose"][0], "nose")
        self.assertEqual(snap, {"v": 1, "type": "snapshot", "floor": "Ann", "lines": [
            {"id": 1, "speaker": "Ann", "text": "earlier", "final": True, "role": ROLE_FLOOR}]})
        self.assertEqual(line, {"v": 1, "type": "line", "id": 2, "speaker": "Bob",
                                "text": "hi there", "final": False, "role": ROLE_SIDE})

    def test_malformed_and_unknown_client_messages_are_ignored(self):
        async def scenario():
            feed = CaptionLineFeed(lambda m: None)
            async with CaptionServer(feed, host="127.0.0.1", port=0) as server:
                async with connect(f"ws://127.0.0.1:{server.port}") as ws:
                    await ws.recv()  # snapshot
                    for junk in ("not json", "[1,2]", '{"type":"bogus"}', b"\xff\x00", "null"):
                        await ws.send(junk)
                    feed.interim(1, "Ann", "still alive")
                    await ws.recv()  # floor
                    return json.loads(await ws.recv())

        self.assertEqual(run(scenario())["text"], "still alive")

    def test_finished_ack_from_a_client_releases_the_next_sign(self):
        from collections import namedtuple
        Gloss = namedtuple("Gloss", "gloss start end")
        frame = [0.1] + [0.0] * 257

        async def scenario():
            feed = CaptionLineFeed(lambda m: None, translate=lambda t: [Gloss(f"g{i}", 0, 1) for i in range(3)],
                                   load_motion=lambda g: [frame] * 3)
            async with CaptionServer(feed, host="127.0.0.1", port=0) as server:
                async with connect(f"ws://127.0.0.1:{server.port}") as ws:
                    await ws.recv()
                    feed.final(1, "Ann", "x")
                    msgs = [json.loads(await ws.recv()) for _ in range(4)]  # floor, line, sign 1, sign 2
                    await ws.send(json.dumps({"v": 1, "type": "ack", "seq": 1, "event": "finished"}))
                    return msgs, json.loads(await ws.recv())

        msgs, third = run(scenario())
        self.assertEqual([m["seq"] for m in msgs[2:]], [1, 2])
        self.assertEqual(third["seq"], 3)

    def test_late_joiner_gets_the_signs_still_in_flight_after_the_snapshot(self):
        from collections import namedtuple
        Gloss = namedtuple("Gloss", "gloss start end")
        frame = [0.1] + [0.0] * 257

        async def scenario():
            feed = CaptionLineFeed(lambda m: None, translate=lambda t: [Gloss(f"g{i}", 0, 1) for i in range(3)],
                                   load_motion=lambda g: [frame] * 3)
            async with CaptionServer(feed, host="127.0.0.1", port=0) as server:
                feed.final(1, "Ann", "x")
                async with connect(f"ws://127.0.0.1:{server.port}") as ws:
                    return [json.loads(await ws.recv()) for _ in range(3)]

        snap, first, second = run(scenario())
        self.assertEqual(snap["type"], "snapshot")
        self.assertEqual([(m["type"], m["seq"]) for m in (first, second)], [("sign", 1), ("sign", 2)])


if __name__ == "__main__":
    unittest.main()
