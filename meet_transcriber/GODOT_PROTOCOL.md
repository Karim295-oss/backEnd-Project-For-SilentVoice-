# Godot caption protocol (v1)

Run the transcriber with `--godot` (optionally `--godot-port 8765`). It listens on all
interfaces, default port **8765**, and streams text WebSocket frames, each a JSON object with
`"v": 1`. Without `--godot` nothing changes (stdout / `--format jsonl` / `--out` are untouched).

```
python meet_transcriber.py https://meet.google.com/abc-defg-hij --godot
python fake_godot_client.py ws://localhost:8765     # prints what Godot would receive
```

## Messages (server -> client)

**snapshot** - sent once on connect: the current `floor` holder (a speaker name, or `null`) and the
last 10 caption lines at their latest text, oldest first.

```json
{"v": 1, "type": "snapshot", "floor": "Ann",
 "lines": [{"id": 4, "speaker": "Ann", "text": "How are you?", "final": true, "role": "floor"}],
 "skeleton": {"pose": ["nose", "left_eye", "..."], "hand": ["wrist", "thumb_cmc", "..."]}}
```

`skeleton` names the joints, in order, of every `sign` frame: 13 `pose` joints (nose, eyes, ears,
shoulders, elbows, wrists, hips) and 21 `hand` joints (the same names for both hands).

A client that connects while signs are in flight receives them right after the snapshot (same
`sign` messages, oldest first); order by `seq` and ignore a `seq` you already have.

**line** - a caption line was created or revised (in-progress text is sent at most every 100 ms).

```json
{"v": 1, "type": "line", "id": 5, "speaker": "Bob", "text": "I am fine", "final": false, "role": "side"}
```

- `role` is `"floor"` (the floor holder's speech) or `"side"` (a side comment). It is fixed when the
  line is created and repeated on every message for that line.
- `id` is unique for the whole session (never reused, even if Meet reuses the same caption element for new speech).
- `text` is the line's **full current text**: replace what you show for that `id`. A long
  utterance may be split into several lines, each locked in turn.
- A line sent just before a client's snapshot can appear again inside it; replace by `id`.
- `final: true` locks the line. Later speech from the same Meet caption element arrives under a **new** `id`. A line is also
  locked when its caption element goes away or is taken over by another speaker.

**floor** - the floor holder changed. One speaker at a time holds the floor; the first speaker with
a new caption line takes it (this message precedes that line). The holder keeps it while their signs
are queued or playing, while they have a caption line still open, and until 2.0 s after their last
caption text or their last sign finishing; then `speaker` is `null`. A line's `role` therefore always
matches who held the floor when it was created, and a `floor` line is always signed when it locks.

```json
{"v": 1, "type": "floor", "speaker": "Ann"}
```

**sign** - sent right after a line with `role: "floor"` is locked (`final: true`): one message per signable gloss of
that line's text, in ASL gloss order. Each gloss is resolved against the sign library: the gloss itself, a manual
synonym, a WordNet synonym in the library, else it is fingerspelled (`kind: "fingerspell"`: one sign made of the
word's letters, 10 frames each; letters missing from the library are skipped). A gloss with nothing signable
produces no message. Side comments are text only and are never signed, even
after the floor is released.

```json
{"v": 1, "type": "sign", "seq": 12, "line": 5, "gloss": "coffee", "kind": "word",
 "span": [7, 13], "fps": 30, "speed": 1,
 "frames": [{"pose": [[0, 0, 0], "..."], "left_hand": null, "right_hand": [[0.1, -0.2, 0.4], "..."]}]}
```

- `seq` increases by one for every sign message of the session; `line` is the caption line's `id`.
- Signs are paced by the server: at most **two** are in flight (the one playing and the next), so
  chain them back-to-back. The next is sent when one finishes. `speed` (1.0-1.75) is how much faster
  than `fps` to play it: it rises when signs are queued behind it. No sign is dropped.
- `span` is `[start, end)` character offsets of the source word in that line's final `text`.
- Coordinates are Godot axes (+Y up, +X signer's left, +Z toward the viewer), nose at the origin,
  rounded to 3 decimals. Hand depth is re-based on the pose wrist of the same side.
- `pose`, `left_hand` and `right_hand` are `null` for a frame where that part was not detected.

## Client -> server

**ack** - report playback of a `sign`:

```json
{"v": 1, "type": "ack", "seq": 12, "event": "started"}
```

`event` is `started` or `finished`. A `finished` ack releases the next sign. Without one, the
server counts a sign finished once its expected end (`frames / fps / speed` after it starts) plus
1.0 s has passed, so playback keeps moving with no client or a stalled one. Acks for unknown
`seq`s are ignored, and so is a `finished` ack for anything but the oldest sign in flight
(signs play in order).

Anything else a client sends (malformed, unknown) is ignored and the connection stays open.
Ignore unknown `type` values and extra fields so the protocol can grow.

## LAN discovery

So the phone app can connect without typing an address, the PC broadcasts a UDP datagram to
`255.255.255.255:8766` once a second while the server runs:

```json
{"v": 1, "service": "silentvoice-captions", "ws": "ws://192.168.1.20:8765"}
```

`ws` uses the PC's LAN IP and the server's port; connect to it as is. Ignore datagrams whose
`service` is not `silentvoice-captions` or whose `v` you don't know. Some networks (guest Wi-Fi,
client isolation) block broadcasts: start the transcriber with `--no-broadcast` to turn it off, or
type the address by hand. The LAN WebSocket URL is always logged at startup
(`godot_server_listening ... url=ws://...`). Allow UDP 8766 inbound on the phone side only if its
firewall requires it; the PC only sends.

## Windows Firewall

Allow inbound connections to the port (run in an elevated PowerShell):

```powershell
New-NetFirewallRule -DisplayName "SILENTVOICE captions" -Direction Inbound -Protocol TCP -LocalPort 8765 -Action Allow
```

Remove it later with `Remove-NetFirewallRule -DisplayName "SILENTVOICE captions"`.
