"""Server-side pacing of sign playback (ADR 0002).

Signs wait in a queue and at most MAX_IN_FLIGHT are with the client at once: the one playing
and the next, so the client can chain them without a gap. A sign is finished when the client
acks it or when its expected end plus GRACE_SECONDS passes, so playback keeps moving with no
client connected. Each sign is dispatched at a speed that catches up on the backlog; none is
dropped. Time is injected (`now`), never slept on.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

MAX_IN_FLIGHT = 2
GRACE_SECONDS = 1.0
CATCH_UP_SECONDS = 3.0
MAX_SPEED = 1.75


@dataclass
class _InFlight:
    seq: int
    message: dict
    play: float                     # seconds the sign takes at its dispatched speed
    sent_at: float
    started_at: float | None = None  # from the client's `started` ack
    not_before: float = 0.0          # the sign before it finished at this time


class SignPlayer:
    def __init__(self, send, now):
        """`send(msg)` delivers a sign message to clients; `now()` is the clock in seconds."""
        self._send, self._now = send, now
        self._queue: deque[dict] = deque()
        self._in_flight: list[_InFlight] = []

    def enqueue(self, signs: list[dict]):
        """Queue signs together, so the first one's speed already sees the rest of the batch."""
        self._queue.extend(signs)
        self._pump()

    def idle(self) -> bool:
        """True when no sign is queued or in flight."""
        return not self._queue and not self._in_flight

    def in_flight_signs(self) -> list[dict]:
        """The sign messages sent but not yet finished, oldest first (for late joiners)."""
        return [e.message for e in self._in_flight]

    def handle_message(self, msg):
        """A decoded client message. Only well-formed acks do anything: `started` for any
        in-flight sign, `finished` only for the oldest one (signs play in order). Anything else
        (malformed, unknown type or seq, out-of-order or repeated) is ignored."""
        if not isinstance(msg, dict) or msg.get("type") != "ack":
            return
        seq, event = msg.get("seq"), msg.get("event")
        if type(seq) is not int:
            return
        entry = next((e for e in self._in_flight if e.seq == seq), None)
        if entry is None:
            return
        if event == "started":
            if entry.started_at is None:  # a repeated ack must not keep pushing the timeout back
                entry.started_at = self._now()
        elif event == "finished" and entry is self._in_flight[0]:
            self._finish(entry, self._now())
            self._pump()

    def tick(self):
        """Finish every in-flight sign whose expected end plus the grace has passed."""
        now = self._now()
        prev_end = 0.0
        for entry in list(self._in_flight):
            start = entry.started_at
            if start is None:
                start = max(entry.sent_at, entry.not_before, prev_end)
            prev_end = start + entry.play
            if now >= prev_end + GRACE_SECONDS:
                self._finish(entry, prev_end)
        self._pump()

    def _finish(self, entry: _InFlight, ended_at: float):
        """The next sign cannot have started before this one ended."""
        self._in_flight.remove(entry)
        for other in self._in_flight:
            other.not_before = max(other.not_before, ended_at)

    def _pump(self):
        while self._queue and len(self._in_flight) < MAX_IN_FLIGHT:
            sign = self._queue.popleft()
            backlog = _duration(sign) + sum(_duration(s) for s in self._queue)
            speed = round(min(max(backlog / CATCH_UP_SECONDS, 1.0), MAX_SPEED), 3)
            sign = {**sign, "speed": speed}
            self._in_flight.append(_InFlight(sign["seq"], sign, _duration(sign) / speed, self._now()))
            self._send(sign)


def _duration(sign: dict) -> float:
    """Seconds the sign takes at 1x."""
    return len(sign["frames"]) / sign["fps"]
