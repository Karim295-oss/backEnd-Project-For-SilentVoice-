import asyncio
import json
import os
import socket
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import contextlib  # noqa: E402
import io  # noqa: E402
from unittest import mock  # noqa: E402

import meet_transcriber  # noqa: E402
from discovery import DISCOVERY_PORT, Announcer, announcement, lan_ip, ws_url  # noqa: E402


def run(coro):
    return asyncio.run(asyncio.wait_for(coro, 5))


def listener():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    sock.setblocking(False)
    return sock


class DiscoveryTests(unittest.TestCase):
    def test_announcement_names_service_version_and_url(self):
        self.assertEqual(announcement("192.168.1.20", 8765),
                         {"v": 1, "service": "silentvoice-captions", "ws": "ws://192.168.1.20:8765"})

    def test_lan_ip_is_an_ipv4_address(self):
        socket.inet_aton(lan_ip())

    def test_default_port_is_8766(self):
        self.assertEqual(DISCOVERY_PORT, 8766)

    def test_listener_receives_announcements_about_every_interval(self):
        async def scenario():
            sock = listener()
            loop = asyncio.get_running_loop()
            target = sock.getsockname()
            async with Announcer("10.0.0.5", 8765, target=target, interval=0.05):
                first = json.loads((await loop.sock_recvfrom(sock, 1024))[0])
                second = json.loads((await loop.sock_recvfrom(sock, 1024))[0])
            sock.close()
            return first, second

        first, second = run(scenario())
        expected = {"v": 1, "service": "silentvoice-captions", "ws": "ws://10.0.0.5:8765"}
        self.assertEqual(first, expected)
        self.assertEqual(second, expected)

    def test_stops_sending_after_exit(self):
        async def scenario():
            sock = listener()
            loop = asyncio.get_running_loop()
            async with Announcer("10.0.0.5", 8765, target=sock.getsockname(), interval=0.02):
                await loop.sock_recvfrom(sock, 1024)
            await asyncio.sleep(0.05)
            while True:  # drain what was in flight
                try:
                    sock.recvfrom(1024)
                except BlockingIOError:
                    break
            await asyncio.sleep(0.1)
            try:
                sock.recvfrom(1024)
            except BlockingIOError:
                return True
            finally:
                sock.close()
            return False

        self.assertTrue(run(scenario()))


class FakeAnnouncer:
    started = []

    def __init__(self, ip, port):
        self.args = (ip, port)

    async def __aenter__(self):
        FakeAnnouncer.started.append(self.args)
        return self

    async def __aexit__(self, *exc):
        pass


def start_lan_announcement(ip, broadcast):
    """Run the bridge's startup step with a fake announcer; return (started, stderr log)."""
    FakeAnnouncer.started = []
    err = io.StringIO()

    async def scenario():
        async with contextlib.AsyncExitStack() as stack:
            with mock.patch.object(meet_transcriber, "Announcer", FakeAnnouncer), \
                    mock.patch.object(meet_transcriber, "lan_ip", lambda: ip), \
                    contextlib.redirect_stderr(err):
                await meet_transcriber.announce_lan(stack, 8765, broadcast)

    run(scenario())
    return FakeAnnouncer.started, err.getvalue()


class StartupTests(unittest.TestCase):
    def test_ws_url_is_shared_by_announcement(self):
        self.assertEqual(ws_url("1.2.3.4", 9), "ws://1.2.3.4:9")
        self.assertEqual(announcement("1.2.3.4", 9)["ws"], ws_url("1.2.3.4", 9))

    def test_startup_logs_url_and_starts_announcer(self):
        started, log = start_lan_announcement("192.168.1.20", broadcast=True)
        self.assertEqual(started, [("192.168.1.20", 8765)])
        self.assertIn("ws://192.168.1.20:8765", log)
        self.assertNotIn("lan_ip_unreachable", log)

    def test_no_broadcast_flag_disables_announcer_but_still_logs_url(self):
        with mock.patch.object(sys, "argv", ["meet_transcriber.py", "https://meet.google.com/a-b-c",
                                              "--godot", "--no-broadcast"]):
            seen = {}

            async def fake_run(args):
                seen["args"] = args
                return 0

            with mock.patch.object(meet_transcriber, "run", fake_run), self.assertRaises(SystemExit):
                meet_transcriber.main()
        self.assertTrue(seen["args"].no_broadcast)
        started, log = start_lan_announcement("192.168.1.20", broadcast=not seen["args"].no_broadcast)
        self.assertEqual(started, [])
        self.assertIn("ws://192.168.1.20:8765", log)

    def test_loopback_fallback_warns_that_url_is_unreachable(self):
        _, log = start_lan_announcement("127.0.0.1", broadcast=True)
        self.assertIn("lan_ip_unreachable", log)


if __name__ == "__main__":
    unittest.main()
