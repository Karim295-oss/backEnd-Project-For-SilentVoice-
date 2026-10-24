# Godot Integration Guide

How to build the Godot (phone) client that shows live Google Meet captions and signs them with an
avatar. The Python side (`meet_transcriber/`) joins the meeting, builds the captions and sign
motions, and paces playback. Godot is a thin renderer: connect, draw captions, animate skeleton
frames, send acks.

The wire format is the reference in [`meet_transcriber/GODOT_PROTOCOL.md`](../meet_transcriber/GODOT_PROTOCOL.md).
This guide is the how-to; if the two ever disagree, the protocol file wins.

## Architecture

```
Google Meet captions
        |
 meet_transcriber.py  --godot       (PC)
   caption tracker -> floor holder -> ASL gloss -> sign library -> paced sign queue
        |  WebSocket :8765  (JSON text frames, "v": 1)
        |  UDP broadcast :8766 (so the phone finds the PC)
        v
 Godot client (Android phone, or desktop for development)
   caption list + avatar renderer + acks
```

Why Godot stays thin: the server owns the floor holder, the sign queue and playback speed
(see [ADR 0002](adr/0002-server-owns-sign-playback.md)). Side comments are text only and never
signed ([ADR 0003](adr/0003-side-comments-never-signed.md)).

## 1. Run the server

From `meet_transcriber/`:

```bash
python meet_transcriber.py https://meet.google.com/abc-defg-hij --godot
```

| Flag | Effect |
|------|--------|
| `--godot` | Start the WebSocket caption server (off by default; other output is unchanged) |
| `--godot-port N` | WebSocket port, default `8765` |
| `--no-broadcast` | Do not announce the server on the LAN (UDP 8766) |

At startup it logs `godot_server_listening ... url=ws://<lan-ip>:8765`. The sign library must exist
(`python scripts/build_speech_sign_library.py --workers 2` from the repo root); otherwise words
fall back to `models/MP_Data_Norm/`.

**Test without Godot:** `python fake_godot_client.py ws://localhost:8765` prints every message.

**Windows Firewall:** allow inbound TCP 8765 from an elevated PowerShell:

```powershell
New-NetFirewallRule -DisplayName "SILENTVOICE captions" -Direction Inbound -Protocol TCP -LocalPort 8765 -Action Allow
```

## 2. Find the server (LAN discovery)

While the server runs it broadcasts this UDP datagram to `255.255.255.255:8766` once a second:

```json
{"v": 1, "service": "silentvoice-captions", "ws": "ws://192.168.1.20:8765"}
```

Listen on UDP 8766, ignore anything whose `service` is not `silentvoice-captions` or whose `v` you
don't know, and connect to `ws` as is. Always offer a manual "enter address" field as well: guest
Wi-Fi and client isolation block broadcasts.

```gdscript
# discovery.gd
extends Node
signal found(ws_url: String)

var _udp := PacketPeerUDP.new()

func _ready() -> void:
    _udp.set_broadcast_enabled(true)
    if _udp.bind(8766) != OK:
        push_warning("UDP 8766 busy; use manual address entry")

func _process(_dt: float) -> void:
    while _udp.get_available_packet_count() > 0:
        var msg = JSON.parse_string(_udp.get_packet().get_string_from_utf8())
        if msg is Dictionary and msg.get("v") == 1 and msg.get("service") == "silentvoice-captions":
            found.emit(msg["ws"])
            _udp.close()
            return
```

On Android, the app needs the `INTERNET` and `ACCESS_NETWORK_STATE` permissions (enable them in the
export preset). Some devices drop broadcast packets while the screen is off or in battery saver.

## 3. Connect and read messages

Every server frame is a text WebSocket frame holding one JSON object with `"v": 1`. Dispatch on
`type`; **ignore unknown types and extra fields** so the protocol can grow.

```gdscript
# caption_client.gd
extends Node

signal snapshot(msg: Dictionary)
signal line(msg: Dictionary)
signal floor_changed(speaker)       # String or null
signal sign(msg: Dictionary)

var _ws := WebSocketPeer.new()
var _open := false

func connect_to(url: String) -> void:
    _ws.connect_to_url(url)

func _process(_dt: float) -> void:
    _ws.poll()
    match _ws.get_ready_state():
        WebSocketPeer.STATE_OPEN:
            _open = true
            while _ws.get_available_packet_count() > 0:
                _handle(_ws.get_packet().get_string_from_utf8())
        WebSocketPeer.STATE_CLOSED:
            if _open:
                _open = false
                # reconnect: wait a second, then connect_to() again

func _handle(text: String) -> void:
    var msg = JSON.parse_string(text)
    if not (msg is Dictionary) or msg.get("v") != 1:
        return
    match msg.get("type"):
        "snapshot": snapshot.emit(msg)
        "line": line.emit(msg)
        "floor": floor_changed.emit(msg.get("speaker"))
        "sign": sign.emit(msg)

func send_ack(seq: int, event: String) -> void:
    _ws.send_text(JSON.stringify({"v": 1, "type": "ack", "seq": seq, "event": event}))
```

On every (re)connect the server sends a `snapshot` first, then any `sign` messages still in flight,
oldest first. Treat the snapshot as authoritative: rebuild your caption list from it.

## 4. Captions

- `snapshot`: `floor` (speaker or `null`), up to 10 `lines` (oldest first), and `skeleton`.
- `line`: a line was created or revised. `text` is the **full current text**, so replace what you
  show for that `id`; never append.
- `id` is unique for the whole session and is never reused. A line sent just before a snapshot may
  show up again inside it: always upsert by `id`.
- `final: true` locks the line. Later speech from the same Meet caption arrives under a new `id`.
- `role`: `"floor"` (the floor holder, will be signed) or `"side"` (side comment, text only). It is
  fixed per line. Suggest styling side comments smaller or dimmer.
- `floor` message: the floor holder changed (`speaker` may be `null`). Show who is "speaking".

```gdscript
var lines := {}          # id -> Dictionary
var order: Array[int] = []

func upsert_line(m: Dictionary) -> void:
    var id := int(m["id"])
    if not lines.has(id):
        order.append(id)
    lines[id] = m
    # trim `order` to the last ~50 and refresh the UI
```

JSON numbers parse as `float` in GDScript; cast `id`, `seq` and the `span` offsets with `int()`.

## 5. Signs and the avatar

A `sign` message arrives for each signable gloss once a floor line is locked:

```json
{"v": 1, "type": "sign", "seq": 12, "line": 5, "gloss": "coffee", "kind": "word",
 "span": [7, 13], "fps": 30, "speed": 1,
 "frames": [{"pose": [[x,y,z], ...], "left_hand": [[x,y,z], ...], "right_hand": null}]}
```

- `kind` is `"word"` or `"fingerspell"` (one sign made of the word's letters; fingerspelled
  sequences are not 30 frames, so always iterate `frames.size()`).
- `span` is `[start, end)` into the line's final `text`: use it to highlight the word being signed.
- Coordinates are already Godot axes: **+Y up, +X the signer's left, +Z toward the viewer**, nose at
  the origin, rounded to 3 decimals. No axis conversion needed.
- `pose`, `left_hand`, `right_hand` are `null` for a frame where that part was not detected. Hold
  the previous pose for that part (or relax it) rather than snapping to zero.
- `skeleton` (from the snapshot) names the joints in the order of each frame array:
  13 `pose` joints (nose, eyes, ears, shoulders, elbows, wrists, hips) and 21 `hand` joints (the same
  names for both hands). Keep it to map array indexes to your rig.

### Rendering

The simplest renderer draws the skeleton as spheres and lines (a stick figure), which is also the
quickest way to verify the pipeline. For a rigged avatar, drive bones from joint positions with
`Skeleton3D` (set bone poses by aiming each bone at its child joint, or use IK targets on wrists and
fingertips). Joint positions are relative to the nose, so scale the whole figure to your avatar's
height.

### Playback loop

```gdscript
# sign_player.gd
extends Node3D

signal started(seq: int)
signal finished(seq: int)

var queue: Array[Dictionary] = []
var current: Dictionary = {}
var t := 0.0
var seen: Dictionary = {}              # seq -> true, so replays are ignored

func enqueue(msg: Dictionary) -> void:
    var seq := int(msg["seq"])
    if seen.has(seq):
        return
    seen[seq] = true
    queue.append(msg)
    queue.sort_custom(func(a, b): return a["seq"] < b["seq"])

func _process(dt: float) -> void:
    if current.is_empty():
        if queue.is_empty():
            return
        current = queue.pop_front()
        t = 0.0
        started.emit(int(current["seq"]))
    t += dt * float(current["speed"]) * float(current["fps"])
    var frames: Array = current["frames"]
    var i := int(t)
    if i >= frames.size():
        var done := int(current["seq"])
        current = {}
        finished.emit(done)
        return
    apply_frame(frames[i])

func apply_frame(frame: Dictionary) -> void:
    # frame["pose"], frame["left_hand"], frame["right_hand"] may each be null
    pass
```

Wire it up: `caption_client.sign` -> `enqueue`, `started` / `finished` -> `send_ack(seq, "started" | "finished")`.

Rules the player must follow:

1. Play signs **in `seq` order, back-to-back**. The server keeps at most two in flight (the one
   playing and the next), so there is always one queued to chain into without a gap.
2. Honour `speed` (1.0 to 1.75): it rises when signs queue up behind the current one.
3. Order by `seq` and ignore a `seq` you already have (late join and reconnect can resend).
4. Optionally interpolate between frames for smoother motion; the data is 30 fps.

## 6. Acks

Send two acks per sign: `{"v":1,"type":"ack","seq":12,"event":"started"}` when it begins and
`"finished"` when it ends. A `finished` ack releases the next sign. If a client never acks (or
stalls), the server counts the sign finished at `frames / fps / speed` after it started, plus 1 s, so
playback keeps moving. Consequences:

- Ack promptly and accurately; late `finished` acks only slow the queue.
- Only the oldest in-flight sign's `finished` counts; acks for unknown `seq` values are ignored.
- The server honours only the first `started` ack for a `seq`; do not resend.

Malformed or unknown client messages are ignored and the connection stays open.

## 7. Reconnecting and late join

The phone sleeps, Wi-Fi drops, the app backgrounds. On reconnect:

1. Reconnect with the last known address (or rediscover over UDP).
2. Handle `snapshot`: replace your caption list and floor holder.
3. Handle the in-flight `sign` messages that follow: dedupe by `seq`.
4. Resume acking.

Signs already played before you reconnected are not replayed.

## 8. Development checklist

1. Run the server and the fake client; confirm captions and `sign` messages print.
2. Build the Godot client against the fake feed first (record its output to a file and replay it, or
   point at a live meeting).
3. Draw a stick-figure skeleton and confirm +Y up and the signer's left is +X.
4. Verify acks: kill the client mid-sign and check the server keeps moving.
5. Test reconnect, `--no-broadcast` with manual address entry, and a phone on the same Wi-Fi.

## Troubleshooting

| Symptom | Likely cause |
|---------|--------------|
| Phone never finds the PC | Guest Wi-Fi or client isolation blocks broadcasts: enter `ws://<pc-ip>:8765` by hand (the URL is logged at startup) |
| Connection refused / times out | Windows Firewall rule missing for TCP 8765, or phone on a different network |
| Discovery shows `127.0.0.1` | The PC has no network; the server warns when it falls back to loopback |
| Captions show but no signs | Speaker was not the floor holder (`role: "side"`), or the word has no sign, synonym or letters |
| Signs stutter or queue up | Late acks; send `finished` as soon as the last frame is shown |
| Avatar mirrored | Remember +X is the signer's left, not the viewer's |
