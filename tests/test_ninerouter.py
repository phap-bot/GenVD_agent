from __future__ import annotations

import unittest
from unittest.mock import patch, MagicMock

from utils.ninerouter import is_9router_running, ensure_9router_running, _find_9router_command


class TestNineRouterService(unittest.TestCase):

    @patch("utils.ninerouter.urlopen")
    def test_is_9router_running_true(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        self.assertTrue(is_9router_running("http://localhost:20128/v1"))

    @patch("utils.ninerouter.urlopen", side_effect=Exception("Connection refused"))
    def test_is_9router_running_false(self, mock_urlopen):
        self.assertFalse(is_9router_running("http://localhost:20128/v1"))

    @patch("utils.ninerouter.is_9router_running", return_value=True)
    def test_ensure_9router_already_running(self, mock_is_running):
        self.assertTrue(ensure_9router_running())

    @patch("shutil.which")
    def test_find_9router_command_direct(self, mock_which):
        mock_which.side_effect = lambda name: "C:\\npm\\9router.cmd" if name == "9router" else None
        cmd = _find_9router_command("http://localhost:20128/v1")
        self.assertEqual(cmd, ["C:\\npm\\9router.cmd", "-n", "-p", "20128"])

    @patch("shutil.which")
    def test_find_9router_command_npx_fallback(self, mock_which):
        mock_which.side_effect = lambda name: "C:\\npm\\npx.cmd" if name == "npx" else None
        cmd = _find_9router_command("http://localhost:20128/v1")
        self.assertEqual(cmd, ["C:\\npm\\npx.cmd", "-y", "9router", "-n", "-p", "20128"])


if __name__ == "__main__":
    unittest.main()
