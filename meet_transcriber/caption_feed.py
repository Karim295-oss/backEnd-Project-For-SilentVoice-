"""Caption lines for the Godot client protocol (v1).

A feed is a transcript Sink: the tracker's interim()/final() calls become `line`
messages passed to `emit`. Meet's caption sources (its on-screen caption elements) are
identified by `source_id`; a source can be reused after its line ends, so line ids are
allocated here and stay unique for the session.

One speaker at a time holds the floor. A line's role (ROLE_FLOOR or ROLE_SIDE) is fixed when it is
created, and only floor lines are signed (ADR 0003). The floor is kept while the holder has an
open line, queued or playing signs, or has made caption text or finished a sign in the last
FLOOR_SILENCE_SECONDS.
"""
from __future__ import annotations

import logging
import time
from collections import OrderedDict

from sign_frames import HAND_JOINTS, POSE_JOINTS, motion_to_frames
from sign_player import SignPlayer

PROTOCOL_VERSION = 1
SNAPSHOT_LINES = 10
DEFAULT_PORT = 8765
log = logging.getLogger(__name__)

SIGN_FPS = 30
FLOOR_SILENCE_SECONDS = 2.0
ROLE_FLOOR = "floor"
ROLE_SIDE = "side"


class CaptionLineFeed:
    def __init__(self, emit, translate=None, load_motion=None, now=time.monotonic,
                 resolve=None):
        """`translate(text)` returns gloss tokens (gloss, start, end) in ASL order and
        `load_motion(gloss)` the gloss's 258-feature sign motion or None; with both, every floor-holder
        final line is followed by one `sign` message per signable gloss. Without both, no
        signs are sent. `resolve(gloss)` may replace `load_motion`: it returns `(kind, motion)`
        (kind "word" or "fingerspell") or None. Signs are paced by a SignPlayer driven by `tick()` and the clients'
        `ack()`s; `now()` is its clock."""
        self.emit = emit
        if resolve is None and load_motion is not None:
            def resolve(gloss):
                return "word", load_motion(gloss)
        self._translate, self._resolve = translate, resolve
        self._signs_enabled = translate is not None and resolve is not None
        self._now = now
        self._player = SignPlayer(lambda msg: self.emit(msg), now)
        self._floor: str | None = None
        self._floor_activity = 0.0  # when the holder last produced caption text or finished signing
        self._player_busy = False
        self._next_seq = 1
        self._next_id = 1
        self._open_line_ids: dict[int, int] = {}  # source id -> id of its open caption line
        self._roles: dict[int, str] = {}  # open line id -> ROLE_FLOOR | ROLE_SIDE, fixed at creation
        self._recent: OrderedDict[int, dict] = OrderedDict()  # line id -> latest message

    def interim(self, source_id: int, speaker: str, text: str):
        """The source's open line (created if needed) now reads `text`."""
        self._line(self._open_line_id(source_id), speaker, text, final=False)

    def final(self, source_id: int, speaker: str, text: str):
        """Lock the source's open line with `text`; later text opens a new line."""
        self._line(self._open_line_id(source_id), speaker, text, final=True)
        del self._open_line_ids[source_id]

    def source_ended(self, source_id: int):
        """The source sends no more text (removed, or reused by another speaker): lock its
        open line as it last read, so a reused source never revises a stale line."""
        line_id = self._open_line_ids.pop(source_id, None)
        if line_id is not None:
            last = self._recent[line_id]
            self._line(line_id, last["speaker"], last["text"], final=True)

    def client_message(self, msg):
        """A decoded message from a client; acks pace the signs, anything else is ignored."""
        self._player.handle_message(msg)
        self._note_player_state()

    def in_flight_signs(self) -> list[dict]:
        """Signs already sent and still playing, for a client that connects mid-sign."""
        return self._player.in_flight_signs()

    def tick(self):
        """Advance sign playback by timeouts; call regularly."""
        self._player.tick()
        self._note_player_state()
        if (self._floor is not None and self._player.idle()
                and ROLE_FLOOR not in self._roles.values()  # the holder's line is still open
                and self._now() - self._floor_activity >= FLOOR_SILENCE_SECONDS):
            self._set_floor(None)

    def snapshot(self) -> dict:
        lines = [{k: m[k] for k in ("id", "speaker", "text", "final", "role")}
                 for m in self._recent.values()]
        return {"v": PROTOCOL_VERSION, "type": "snapshot", "floor": self._floor, "lines": lines,
                "skeleton": {"pose": POSE_JOINTS, "hand": HAND_JOINTS}}

    def _open_line_id(self, source_id: int) -> int:
        if source_id not in self._open_line_ids:
            self._open_line_ids[source_id] = self._next_id
            self._next_id += 1
        return self._open_line_ids[source_id]

    def _line(self, line_id: int, speaker: str, text: str, final: bool):
        if line_id not in self._roles:  # a new line: the first speaker takes a free floor
            if self._floor is None:
                self._set_floor(speaker)
            self._roles[line_id] = ROLE_FLOOR if speaker == self._floor else ROLE_SIDE
        role = self._roles[line_id]
        if role == ROLE_FLOOR:  # the floor cannot change while a floor line is open
            self._floor_activity = self._now()
        if final:
            del self._roles[line_id]  # roles are only needed while a line can still be revised
        msg = {"v": PROTOCOL_VERSION, "type": "line", "id": line_id,
               "speaker": speaker, "text": text, "final": final, "role": role}
        self._recent[line_id] = msg
        while len(self._recent) > SNAPSHOT_LINES:
            self._recent.popitem(last=False)
        self.emit(msg)
        if final and self._signs_enabled and role == ROLE_FLOOR:
            self._emit_signs(line_id, text)
            self._note_player_state()

    def _note_player_state(self):
        """The holder's silence also counts from the moment their last sign finished."""
        busy = not self._player.idle()
        if self._player_busy and not busy:
            self._floor_activity = self._now()
        self._player_busy = busy

    def _set_floor(self, speaker: str | None):
        self._floor = speaker
        if speaker is not None:
            self._floor_activity = self._now()
        self.emit({"v": PROTOCOL_VERSION, "type": "floor", "speaker": speaker})

    def _emit_signs(self, line_id: int, text: str):
        """Sign failures never break the caption line: a bad gloss is logged and skipped."""
        try:
            tokens = self._translate(text)
        except Exception:
            log.warning("translating line %d failed; sending no signs", line_id, exc_info=True)
            return
        signs = []
        for token in tokens:
            try:
                resolved = self._resolve(token.gloss)
                kind, motion = resolved if resolved else (None, None)
                frames = motion_to_frames(motion) if _has_detections(motion) else None
            except Exception:
                log.warning("sign motion for %r failed; skipping it", token.gloss, exc_info=True)
                continue
            if frames is None:
                continue
            signs.append({"v": PROTOCOL_VERSION, "type": "sign", "seq": self._next_seq,
                          "line": line_id, "gloss": token.gloss, "kind": kind,
                          "span": [token.start, token.end], "fps": SIGN_FPS,
                          "frames": frames})
            self._next_seq += 1
        self._player.enqueue(signs)


def _has_detections(motion) -> bool:
    """False for a missing, empty or all-zero motion (nothing was ever detected)."""
    return bool(motion) and any(any(frame) for frame in motion)
