"""End-to-end check of OBSERVER_JS + CaptionTracker against a fake captions page.
Skipped when Playwright/Chromium isn't installed."""
import asyncio
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from meet_transcriber import OBSERVER_JS, CaptionTracker  # noqa: E402

try:
    from playwright.async_api import async_playwright
except ImportError:  # pragma: no cover
    async_playwright = None

PAGE = """<body><div id=other>video tiles</div>
<div role="region" aria-label="Captions"><div id=c></div></div></body>"""

ADD_BLOCK = """([speaker, text]) => {
  const d = document.createElement('div');
  d.innerHTML = '<img src="data:image/gif;base64,R0lGODlhAQABAAAAACw="><span>' + speaker + '</span><div><span>' + text + '</span></div>';
  document.getElementById('c').appendChild(d);
}"""


class Sink:
    def __init__(self):
        self.finals = []

    def final(self, block, speaker, text):
        self.finals.append((speaker, text))

    def interim(self, *a):
        pass


@unittest.skipIf(async_playwright is None, "playwright not installed")
class ObserverTest(unittest.TestCase):
    def test_observer_feeds_tracker(self):
        async def go():
            sink = Sink()
            tracker = CaptionTracker(sink, idle=0.5)
            async with async_playwright() as p:
                try:
                    browser = await p.chromium.launch()
                except Exception as e:
                    self.skipTest(f"chromium unavailable: {str(e)[:80]}")
                page = await browser.new_page()
                await page.expose_function("onCaptions", lambda s: tracker.update(json.loads(s)))
                await page.set_content(PAGE)
                await page.evaluate(OBSERVER_JS)
                await page.evaluate(ADD_BLOCK, ["Ann", "Hello there."])
                await page.evaluate("document.getElementById('other').textContent = 'noise'")
                await asyncio.sleep(0.2)
                tracker.tick()
                await asyncio.sleep(0.8)
                tracker.tick()
                await browser.close()
            return sink.finals

        self.assertEqual(asyncio.run(go()), [("Ann", "Hello there.")])


if __name__ == "__main__":
    unittest.main()
