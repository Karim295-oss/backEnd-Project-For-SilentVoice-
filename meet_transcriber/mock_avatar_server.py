"""Mock SILENTVOICE avatar server (GODOT_PROTOCOL v1) for testing Godot without Google Meet.

Run:  python mock_avatar_server.py
Godot connects to ws://127.0.0.1:8765, gets a snapshot, then a looping "wave" sign.
"""
import asyncio
import json
import math

try:
    from websockets.asyncio.server import serve
except ImportError:  # older websockets
    from websockets import serve

PORT = 8765
FPS = 30

POSE = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip"]
FINGERS = ["thumb", "index_finger", "middle_finger", "ring_finger", "pinky"]
HAND = ["wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip"]
for f in FINGERS[1:]:
    HAND += [f"{f}_mcp", f"{f}_pip", f"{f}_dip", f"{f}_tip"]


def make_frame(t: float) -> dict:
    """Godot axes: +Y up, +X signer's left, nose at origin. Signer's right arm waves."""
    wave = math.sin(t * 2 * math.pi * 2)  # 2 waves per second
    joints = {
        "nose": (0, 0, 0),
        "left_eye": (0.03, 0.03, 0.05), "right_eye": (-0.03, 0.03, 0.05),
        "left_ear": (0.08, 0, -0.02), "right_ear": (-0.08, 0, -0.02),
        "left_shoulder": (0.2, -0.3, 0), "right_shoulder": (-0.2, -0.3, 0),
        "left_elbow": (0.25, -0.6, 0), "left_wrist": (0.25, -0.88, 0),
        "right_elbow": (-0.45, -0.3, 0.05),
        "right_wrist": (-0.45 + 0.12 * wave, 0.0, 0.05),
        "left_hip": (0.12, -0.8, 0), "right_hip": (-0.12, -0.8, 0),
    }
    return {
        "pose": [[round(v, 3) for v in joints[name]] for name in POSE],
        "left_hand": None,
        "right_hand": None,
    }


def make_sign(seq: int, gloss: str = "hello", seconds: float = 2.0) -> dict:
    n = int(seconds * FPS)
    return {
        "v": 1, "type": "sign", "seq": seq, "line": 1, "gloss": gloss,
        "kind": "word", "span": [0, len(gloss)], "fps": FPS, "speed": 1,
        "frames": [make_frame(i / FPS) for i in range(n)],
    }


async def wait_for_finished(ws, seq: int, timeout: float):
    """Read client messages until the 'finished' ack for seq (or timeout)."""
    try:
        async with asyncio.timeout(timeout):
            async for raw in ws:
                msg = json.loads(raw)
                print("client ->", msg)
                if msg.get("type") == "ack" and msg.get("seq") == seq and msg.get("event") == "finished":
                    return
    except TimeoutError:
        print(f"no 'finished' ack for seq {seq} (timeout)")


async def handler(ws):
    print("Godot connected")
    snapshot = {"v": 1, "type": "snapshot", "floor": None, "lines": [],
                "skeleton": {"pose": POSE, "hand": HAND}}
    await ws.send(json.dumps(snapshot))

    seq = 0
    while True:
        seq += 1
        sign = make_sign(seq)
        await ws.send(json.dumps(sign))
        print(f"sent sign seq={seq} ({len(sign['frames'])} frames)")
        await wait_for_finished(ws, seq, timeout=len(sign["frames"]) / FPS + 3)
        await asyncio.sleep(1.5)


async def main():
    async with serve(handler, "127.0.0.1", PORT):
        print(f"Mock avatar server listening on ws://127.0.0.1:{PORT}  (Ctrl+C to stop)")
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass