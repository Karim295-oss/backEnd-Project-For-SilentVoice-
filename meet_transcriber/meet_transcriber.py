#!/usr/bin/env python3
"""
Google Meet captions transcriber (participant bot).

Joins a Meet as a visibly named participant, turns on live captions, and prints
transcript events in real time (stdout + optional file).

Install:
    pip install playwright
    playwright install chromium

Run (guest join, headed - most reliable):
    python meet_transcriber.py https://meet.google.com/abc-defg-hij

Live interim updates + machine-readable output (for piping into another tool):
    python meet_transcriber.py URL --live --format jsonl

    jsonl events: {"type": "interim"|"final", "block": 3, "speaker": "...",
                   "text": "...", "ts": "HH:MM:SS"}
    A "final" event carries the text that replaces the block's interim text.

Run headless (Meet may block or degrade this):
    python meet_transcriber.py URL --headless

Attach to a Chrome you started yourself (for a signed-in bot account):
    chrome.exe --remote-debugging-port=9222 --user-data-dir="E:\\meet-bot-profile"
    python meet_transcriber.py URL --cdp http://localhost:9222

Everything in SEL, OBSERVER_JS and ensure_english() is UNVERIFIED against
Meet's current DOM. If captions don't appear, run with --debug and adjust.
This tool does not attempt to evade Google's bot checks.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from caption_feed import DEFAULT_PORT, CaptionLineFeed
from discovery import Announcer, lan_ip, ws_url

EXIT_OK, EXIT_ERROR, EXIT_NOT_ADMITTED, EXIT_BLOCKED, EXIT_CAPTIONS, EXIT_DISCONNECTED = 0, 1, 2, 3, 4, 5

DEFAULT_NOTICE = "This meeting is being translated to sign lang"

# ---------------------------------------------------------------- selectors
# NEEDS VERIFICATION: all of these. Meet changes its UI without notice.
SEL = {
    "name_input": 'input[aria-label="Your name"], input[placeholder="Your name"]',
    "leave": 'button[aria-label="Leave call"]',
    "captions_on": 'button[aria-label="Turn on captions"]',
    "captions_off": 'button[aria-label="Turn off captions"]',
    "mic_off": 'button[aria-label^="Turn off microphone"]',
    "cam_off": 'button[aria-label^="Turn off camera"]',
    "chat_input": 'textarea[aria-label="Send a message"], textarea[placeholder="Send a message"]',
}
# Signed-in accounts already in the call see "Join here too"; icon ligature text may prefix the name.
BTN_JOIN = re.compile(r"(Join now|Ask to join|Join here too)$", re.I)
BTN_NO_MEDIA = re.compile(r"Continue without microphone and camera", re.I)
BTN_GOT_IT = re.compile(r"^Got it$", re.I)
BTN_CHAT = re.compile(r"^Chat with everyone", re.I)
BTN_CAPTIONS = re.compile(r"turn on captions", re.I)

ENDED_TEXTS = ("You left the meeting", "You've been removed", "You were removed",
               "The meeting has ended", "Return to home screen")
DENIED_TEXTS = ("denied your request", "No one responded to your request",
                "You can't join this video call")
CAPTION_HINT_RE = re.compile(r"[^\n]*(caption|transcript)[^\n]*", re.I)

# Watches the captions region and pushes snapshots to Python via window.onCaptions.
# Heuristic: a caption block = innermost <div> inside the Captions region that
# contains an avatar <img> and >= 2 text nodes (first = speaker, rest = text).
OBSERVER_JS = r"""
() => {
  if (window.__capObs) return;
  window.__capObs = true;
  const REGION = 'div[role="region"][aria-label="Captions"]';
  const ids = new WeakMap();
  let nextId = 1, last = "", timer = null;
  const texts = (el) => {
    const w = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
    const out = []; let n;
    while ((n = w.nextNode())) { const t = n.nodeValue.trim(); if (t) out.push(t); }
    return out;
  };
  const snapshot = () => {
    const region = document.querySelector(REGION);
    if (!region) return [];
    const cands = [];
    region.querySelectorAll('div').forEach(d => {
      if (!d.querySelector('img')) return;
      const parts = texts(d);
      if (parts.length >= 2) cands.push([d, parts]);
    });
    const inner = cands.filter(([d]) => !cands.some(([o]) => o !== d && d.contains(o)));
    return inner.map(([d, parts]) => {
      if (!ids.has(d)) ids.set(d, nextId++);
      return { id: ids.get(d), speaker: parts[0], text: parts.slice(1).join(' ') };
    });
  };
  const push = () => {
    const s = JSON.stringify(snapshot());
    if (s !== last) { last = s; window.onCaptions(s); }
  };
  new MutationObserver((muts) => {
    const region = document.querySelector(REGION);
    // Ignore churn elsewhere in the call UI (video tiles, timers, ...).
    if (region && !muts.some(m => region.contains(m.target))) return;
    if (!timer) timer = setTimeout(() => { timer = null; push(); }, 40);
  }).observe(document.body, { childList: true, subtree: true, characterData: true });
  setInterval(push, 500);
}
"""


class GodotSink:
    """Wraps the normal Sink: finals (and interims, if `echo_interim`) go through unchanged,
    and everything is also fed to the Godot caption line feed."""

    def __init__(self, inner, feed, echo_interim: bool):
        self.inner, self.feed, self.echo_interim = inner, feed, echo_interim

    def interim(self, block: int, speaker: str, text: str):
        if self.echo_interim:
            self.inner.interim(block, speaker, text)
        self.feed.interim(block, speaker, text)

    def final(self, block: int, speaker: str, text: str):
        self.inner.final(block, speaker, text)
        self.feed.final(block, speaker, text)

    def block_ended(self, block: int):
        self.feed.source_ended(block)

    def close(self):
        self.inner.close()


class BotError(Exception):
    def __init__(self, code: int, msg: str):
        super().__init__(msg)
        self.code = code


def log(level: str, event: str, **kw):
    rec = {"ts": datetime.now().isoformat(timespec="seconds"), "level": level, "event": event, **kw}
    print(json.dumps(rec, ensure_ascii=False), file=sys.stderr, flush=True)


class Sink:
    """Writes transcript events. Text mode: '[HH:MM:SS] Speaker: text' for finals,
    '~ Speaker: text' for interims (stdout only). jsonl mode: one JSON object per event.
    The --out file only ever receives finals."""

    def __init__(self, path: str | None, fmt: str = "text"):
        self.fmt = fmt
        self.f = open(path, "a", encoding="utf-8") if path else None

    def _line(self, kind: str, block: int, speaker: str, text: str) -> str:
        ts = f"{datetime.now():%H:%M:%S}"
        if self.fmt == "jsonl":
            return json.dumps({"type": kind, "block": block, "speaker": speaker, "text": text, "ts": ts},
                              ensure_ascii=False)
        if kind == "interim":
            return f"~ {speaker}: {text}"
        return f"[{ts}] {speaker}: {text}"

    def interim(self, block: int, speaker: str, text: str):
        print(self._line("interim", block, speaker, text), flush=True)

    def final(self, block: int, speaker: str, text: str):
        line = self._line("final", block, speaker, text)
        print(line, flush=True)
        if self.f:
            self.f.write(line + "\n")
            self.f.flush()

    def close(self):
        if self.f:
            self.f.close()


# A sentence ends at . ! ? … (optionally followed by closing quotes/brackets).
_SENT_BOUNDARY = re.compile(r"[.!?\u2026][\"')\]]*(?=\s+\S)")  # followed by more text
_SENT_END = re.compile(r"[.!?\u2026][\"')\]]*$")                # at end of text


def _norm(word: str) -> str:
    return re.sub(r"[^\w']+", "", word.lower())


class CaptionTracker:
    """Turns Meet's caption blocks into final lines, and (optionally) interim events.

    Meet's caption block is a *sliding window*: it grows word by word, drops its oldest
    words, and keeps revising punctuation/recent words. So instead of prefix matching,
    each block remembers the (normalized) words it has already emitted and every new
    snapshot is aligned against that tail: only words past the overlap are new.
    Text is held back until stable:
      * everything up to a sentence end that the speaker has already moved past, or
      * a trailing sentence end unchanged for `settle` seconds, or
      * anything unchanged for `idle` seconds, or
      * the block disappearing / changing speaker.
    Time is injectable (`now=`) for testing.
    """

    HISTORY = 40  # emitted words remembered per block

    def __init__(self, sink, idle: float = 0.8, settle: float = 0.5,
                 live: bool = False, interim_interval: float = 0.1):
        self.sink, self.idle, self.settle = sink, idle, settle
        self.live, self.interim_interval = live, interim_interval
        self.blocks: dict[int, dict] = {}
        self.seen_any = False

    # -- input -------------------------------------------------------------
    def update(self, snap: list[dict], now: float | None = None):
        now = time.monotonic() if now is None else now
        seen = set()
        for b in snap:
            self.seen_any = True
            bid = b["id"]
            seen.add(bid)
            st = self.blocks.get(bid)
            if st is None:
                self.blocks[bid] = self._new(b, now)
            elif st["speaker"] != b["speaker"]:
                # Meet recycled this node for a different speaker: close out the old one.
                self._commit(bid, st, st["text"])
                self._ended(bid)
                self.blocks[bid] = self._new(b, now)
            elif st["text"] != b["text"]:
                st["text"], st["changed"] = b["text"], now
        for bid in list(self.blocks):
            if bid not in seen:
                st = self.blocks.pop(bid)
                self._commit(bid, st, st["text"])
                self._ended(bid)
        self.tick(now)

    @staticmethod
    def _new(b: dict, now: float) -> dict:
        return {"speaker": b["speaker"], "text": b["text"], "hist": [], "changed": now,
                "interim": "", "interim_t": -1e9}

    # -- time-driven -------------------------------------------------------
    def tick(self, now: float | None = None):
        now = time.monotonic() if now is None else now
        for bid, st in list(self.blocks.items()):
            cut = self._ready_cut(st["text"], st, now)
            if cut:
                self._commit(bid, st, st["text"][:cut])
            if self.live:
                self._interim(bid, st, now)

    def flush_all(self):
        for bid, st in list(self.blocks.items()):
            self._commit(bid, st, st["text"])
            self._ended(bid)

    # -- internals ---------------------------------------------------------
    def _ready_cut(self, text: str, st: dict, now: float) -> int:
        if not self._new_tokens(st, text):
            return 0
        quiet = now - st["changed"]
        if quiet >= self.idle:
            return len(text)
        if quiet >= self.settle and _SENT_END.search(text):
            return len(text)
        last = None
        for last in _SENT_BOUNDARY.finditer(text):
            pass
        return last.end() if last else 0

    def _new_tokens(self, st: dict, text: str) -> list[str]:
        """Words of `text` not yet emitted, found by aligning against the emitted tail."""
        toks = text.split()
        hist = st["hist"]
        if not hist or not toks:
            return toks
        norms = [_norm(t) for t in toks]
        # Whole window already contained in recent history (a revision shrank it).
        w = [n for n in norms[-8:] if n]
        h = [x for x in hist]
        if w and any(h[i:i + len(w)] == w for i in range(len(h) - len(w) + 1)):
            return []
        for m in range(min(len(hist), len(norms)), 0, -1):
            tail = hist[-m:]
            for i in range(len(norms) - m, -1, -1):
                if norms[i:i + m] == tail and (m >= 2 or i == 0):
                    return toks[i + m:]
        return toks

    def _ended(self, bid: int):
        """Tell sinks that care that this block will send no more text, even if its last
        commit emitted no final (e.g. a revision shrank back into already-emitted words)."""
        ended = getattr(self.sink, "block_ended", None)
        if ended:
            ended(bid)

    def _commit(self, bid: int, st: dict, upto: str):
        new = self._new_tokens(st, upto)
        if new:
            st["hist"] = (st["hist"] + [n for n in map(_norm, new) if n])[-self.HISTORY:]
            self.sink.final(bid, st["speaker"], " ".join(new))
        st["interim"] = ""

    def _interim(self, bid: int, st: dict, now: float):
        pending = " ".join(self._new_tokens(st, st["text"]))
        if pending and pending != st["interim"] and now - st["interim_t"] >= self.interim_interval:
            st["interim"], st["interim_t"] = pending, now
            self.sink.interim(bid, st["speaker"], pending)


async def body_text(page) -> str:
    try:
        return await page.evaluate("document.body ? document.body.innerText : ''")
    except Exception:
        return ""


async def caption_hint(page) -> str:
    """Short snippet of any caption/transcript message on screen, for diagnosing failures."""
    m = CAPTION_HINT_RE.search(await body_text(page))
    return m.group(0).strip()[:160] if m else ""


async def click_if_present(page, name_re, timeout=1500, wait: float = 0) -> bool:
    btn = page.get_by_role("button", name=name_re)
    try:
        if wait:  # the join screen renders late; wait for the button instead of sampling once
            await btn.first.wait_for(state="visible", timeout=wait * 1000)
        if await btn.count():
            await btn.first.click(timeout=timeout)
            return True
    except Exception:
        pass
    return False


async def join_meeting(page, url: str, name: str, admit_timeout: float):
    await page.goto(url, wait_until="domcontentloaded")
    await page.wait_for_timeout(3000)
    if "accounts.google.com" in page.url:
        raise BotError(EXIT_BLOCKED, "Meeting requires sign-in. Use --cdp with a signed-in Chrome, "
                                     "or host a meeting that allows guests.")
    for _ in range(3):  # dismiss mic/cam + info prompts
        hit = await click_if_present(page, BTN_NO_MEDIA) or await click_if_present(page, BTN_GOT_IT)
        if not hit:
            break
        await page.wait_for_timeout(500)

    name_box = page.locator(SEL["name_input"])
    if await name_box.count():
        await name_box.first.fill(name)
    for key in ("mic_off", "cam_off"):
        try:
            b = page.locator(SEL[key])
            if await b.count():
                await b.first.click(timeout=1500)
        except Exception:
            pass

    if not await click_if_present(page, BTN_JOIN, timeout=5000, wait=15):
        text = await body_text(page)
        if any(t in text for t in DENIED_TEXTS):
            raise BotError(EXIT_BLOCKED, "Meet refused this browser/meeting (can't join).")
        raise BotError(EXIT_ERROR, "Join button not found (selectors may be outdated).")
    log("info", "join_requested", name=name)

    deadline = time.monotonic() + admit_timeout
    while time.monotonic() < deadline:
        if await page.locator(SEL["leave"]).count():
            log("info", "joined")
            await page.wait_for_timeout(2000)  # let the in-call toolbar finish rendering
            return
        text = await body_text(page)
        if any(t in text for t in DENIED_TEXTS):
            raise BotError(EXIT_NOT_ADMITTED, "Join denied by host.")
        await page.wait_for_timeout(1000)
    raise BotError(EXIT_NOT_ADMITTED, f"Not admitted within {admit_timeout:.0f}s.")


async def captions_are_on(page) -> bool:
    return bool(await page.locator(SEL["captions_off"]).count())


async def enable_captions(page, attempts: int = 5) -> bool:
    """Turn captions on, retrying: right after joining the toolbar can lag behind, and Meet
    sometimes briefly reports captions/transcript as unavailable (a rerun used to fix it)."""
    for i in range(1, attempts + 1):
        if await captions_are_on(page):
            return True
        try:
            await page.mouse.move(600 + i * 10, 600)  # reveal the controls bar
            btn = page.locator(SEL["captions_on"])
            if not await btn.count():
                btn = page.get_by_role("button", name=BTN_CAPTIONS)
            if await btn.count():
                await btn.first.click(timeout=4000)
            elif i > 1:  # shortcut only after the button had a chance to appear
                await page.keyboard.press("c")
        except Exception as e:
            log("warn", "captions_click_failed", attempt=i, error=str(e)[:120])
        await page.wait_for_timeout(1200 * i)
        if await captions_are_on(page):
            return True
        log("warn", "captions_retry", attempt=i, of=attempts, on_screen=await caption_hint(page))
    return False


async def ensure_english(page):
    """Best effort: make sure Meet's caption language ("Meeting language" dropdown in the
    captions bar) is English. Never fatal."""
    try:
        combo = page.get_by_role("combobox", name=re.compile(r"meeting language", re.I)).first
        await combo.wait_for(state="visible", timeout=4000)
        current = (await combo.inner_text(timeout=2000)).strip()
        if re.search(r"english", current, re.I):
            log("info", "language_already_english", current=current[:40])
            return
        await combo.click(timeout=2000)
        await page.get_by_role("option", name=re.compile(r"^English", re.I)).first.click(timeout=3000)
        log("info", "language_set_english", was=current[:40])
    except Exception as e:
        log("warn", "english_failed", error=str(e)[:160])
        try:
            await page.keyboard.press("Escape")
        except Exception:
            pass


async def post_notice(page, text: str):
    try:
        chat = page.get_by_role("button", name=BTN_CHAT).first
        await chat.click(timeout=4000)
        box = page.locator(SEL["chat_input"]).first
        await box.fill(text, timeout=4000)
        await box.press("Enter")
        log("info", "notice_posted")
        await chat.click(timeout=2000)  # close the panel so it doesn't cover the captions
    except Exception as e:
        log("warn", "notice_failed", error=str(e)[:120])


async def leave_and_close(page, owns_browser: bool):
    if page is None:
        return
    try:
        if not page.is_closed():
            leave = page.locator(SEL["leave"])
            if await leave.count():
                await leave.first.click(timeout=3000)
                log("info", "left_meeting")
                await page.wait_for_timeout(500)
            await page.close()
    except Exception as e:
        log("warn", "leave_failed", error=str(e)[:120])


def _sign_sources():
    """(translate, resolve) for the feed's sign messages; loads spaCy, so only with --godot."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
    from grammar_translator import english_to_gloss_tokens
    from sign_motions import load_sign_motion
    from sign_resolver import cached_resolver
    return english_to_gloss_tokens, cached_resolver(load_sign_motion)


async def announce_lan(stack, port: int, broadcast: bool):
    """Log the LAN WebSocket URL and, unless disabled, start announcing it over UDP."""
    ip = lan_ip()
    log("info", "godot_server_listening", port=port, url=ws_url(ip, port))
    if ip.startswith("127."):
        log("warn", "lan_ip_unreachable", ip=ip,
            hint="no LAN address found; the phone cannot reach this URL, check the network")
    if broadcast:
        await stack.enter_async_context(Announcer(ip, port))


async def run(args) -> int:
    sink = Sink(args.out, args.format)
    godot = None
    if args.godot:
        from caption_server import CaptionServer
        translate, resolve = _sign_sources()
        feed = CaptionLineFeed(lambda msg: None, translate, resolve=resolve)
        godot = CaptionServer(feed, port=args.godot_port)
        sink = GodotSink(sink, feed, echo_interim=args.live)
    tracker = CaptionTracker(sink, args.idle, live=args.live or args.godot)
    page = None
    owns_browser = not args.cdp
    try:
        from playwright.async_api import async_playwright
        async with contextlib.AsyncExitStack() as stack:
            if godot:
                await stack.enter_async_context(godot)
                await announce_lan(stack, godot.port, broadcast=not args.no_broadcast)
            p = await stack.enter_async_context(async_playwright())
            if args.cdp:
                browser = await p.chromium.connect_over_cdp(args.cdp)
                ctx = browser.contexts[0] if browser.contexts else await browser.new_context()
            else:
                browser = await p.chromium.launch(headless=args.headless, channel=args.channel)
                ctx = await browser.new_context(viewport={"width": 1280, "height": 800},
                                                locale="en-US")
            page = await ctx.new_page()

            dead = asyncio.Event()
            page.on("crash", lambda *_: (log("error", "page_crashed"), dead.set()))
            page.on("close", lambda *_: dead.set())
            browser.on("disconnected", lambda *_: dead.set())

            def on_caps(payload: str):
                try:
                    tracker.update(json.loads(payload))
                except Exception as e:
                    log("error", "caption_parse_failed", error=str(e))

            await page.expose_function("onCaptions", on_caps)

            try:
                await join_meeting(page, args.url, args.name, args.admit_timeout)
                if not await enable_captions(page):
                    raise BotError(EXIT_CAPTIONS, "Could not enable captions after retries (disabled by "
                                                  "host/admin, or selectors outdated). On screen: "
                                                  + (await caption_hint(page) or "nothing caption-related"))
                await page.evaluate(OBSERVER_JS)
                log("info", "captions_enabled")
                await ensure_english(page)
                if not args.no_notice:
                    await post_notice(page, args.notice)

                started = last_check = last_warn = time.monotonic()
                dumped = False
                while True:
                    try:
                        await asyncio.wait_for(dead.wait(), timeout=0.1)
                        log("error", "page_or_browser_gone")
                        return EXIT_DISCONNECTED
                    except asyncio.TimeoutError:
                        pass
                    now = time.monotonic()
                    tracker.tick(now)
                    if godot:
                        feed.tick()
                    if args.duration and now - started > args.duration:
                        log("info", "duration_reached")
                        return EXIT_OK
                    if args.debug and not dumped and now - started > 45:
                        dumped = True
                        with open("debug_dom.html", "w", encoding="utf-8") as f:
                            f.write(await page.content())
                        log("info", "debug_dom_saved", path="debug_dom.html")
                    if now - started > 60 and not tracker.seen_any and now - last_warn >= 30:
                        last_warn = now
                        log("warn", "no_captions_seen",
                            hint="If people are speaking, update SEL/OBSERVER_JS; run with --debug.")
                    if now - last_check >= 2:
                        last_check = now
                        if not await page.locator(SEL["leave"]).count():
                            text = await body_text(page)
                            if any(t in text for t in ENDED_TEXTS):
                                log("info", "meeting_ended")
                                return EXIT_OK
                        try:
                            if not await page.evaluate("!!window.__capObs"):
                                log("warn", "observer_lost_reinjecting")
                                await page.evaluate(OBSERVER_JS)
                            if not await captions_are_on(page) and await page.locator(SEL["captions_on"]).count():
                                log("warn", "captions_turned_off_reenabling")
                                await enable_captions(page, attempts=3)
                        except Exception as e:
                            log("warn", "watchdog_error", error=str(e)[:120])
            finally:
                tracker.flush_all()
                await leave_and_close(page, owns_browser)
                if owns_browser:
                    try:
                        await browser.close()
                    except Exception:
                        pass

    except BotError as e:
        log("error", "bot_error", code=e.code, message=str(e))
        return e.code
    except (KeyboardInterrupt, asyncio.CancelledError):
        log("info", "interrupted")
        return EXIT_OK
    except Exception as e:
        log("error", "unexpected", error=repr(e))
        return EXIT_ERROR
    finally:
        tracker.flush_all()
        sink.close()


def main():
    ap = argparse.ArgumentParser(description="Google Meet captions transcriber bot")
    ap.add_argument("url", help="Meet URL, e.g. https://meet.google.com/abc-defg-hij")
    ap.add_argument("--name", default="Transcriber Bot", help="display name shown to participants")
    ap.add_argument("--out", help="append final transcript lines to this file")
    ap.add_argument("--format", choices=("text", "jsonl"), default="text", help="output format")
    ap.add_argument("--live", action="store_true", help="also emit interim (in-progress) updates to stdout")
    ap.add_argument("--godot", action="store_true", help="stream caption lines to Godot clients over WebSocket")
    ap.add_argument("--godot-port", type=int, default=DEFAULT_PORT, help="WebSocket port for --godot (default %(default)s)")
    ap.add_argument("--no-broadcast", action="store_true", help="with --godot, don't announce the server on the LAN (UDP port 8766)")
    ap.add_argument("--headless", action="store_true", help="run without a window (Meet may block this)")
    ap.add_argument("--channel", default=None, help='use installed browser, e.g. "chrome" or "msedge"')
    ap.add_argument("--cdp", default=None, help="attach to Chrome started with --remote-debugging-port")
    ap.add_argument("--admit-timeout", type=float, default=180, help="seconds to wait for host admission")
    ap.add_argument("--idle", type=float, default=0.8, help="seconds without change before a line is finalized")
    ap.add_argument("--notice", default=DEFAULT_NOTICE, help="chat message posted after joining")
    ap.add_argument("--no-notice", action="store_true", help="don't post the chat message")
    ap.add_argument("--duration", type=float, default=0, help="leave after this many seconds of transcribing (0 = until the meeting ends)")
    ap.add_argument("--debug", action="store_true", help="save page HTML after 45s to debug_dom.html")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    try:
        code = asyncio.run(run(args))
    except KeyboardInterrupt:
        code = EXIT_OK
    sys.exit(code)


if __name__ == "__main__":
    main()
