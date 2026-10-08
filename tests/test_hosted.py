import os
import unittest
from unittest.mock import patch

from deploy import hosted


class HostedTests(unittest.TestCase):
    def test_host_port_default_override_and_invalid_values(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(hosted.hosted_port(), 10000)
        with patch.dict(os.environ, {"PORT": "7860"}):
            self.assertEqual(hosted.hosted_port(), 7860)
        for value in ("0", "65536", "-1", "abc", " 8000", "１２３"):
            with self.subTest(value=value), patch.dict(os.environ, {"PORT": value}), self.assertRaises(hosted.PreflightError):
                hosted.hosted_port()

    def test_cloud_binds_all_interfaces_through_guarded_native_launcher(self):
        with patch.dict(os.environ, {"PORT": "10001"}), patch.object(hosted, "serve") as serve:
            self.assertEqual(hosted.main(), 0)
        serve.assert_called_once_with(host="0.0.0.0", port=10001)

    def test_failure_does_not_print_underlying_secret(self):
        with patch.object(hosted, "serve", side_effect=RuntimeError("PRIVATE_SECRET")), patch("sys.stderr") as output:
            self.assertEqual(hosted.main(), 1)
        self.assertNotIn("PRIVATE_SECRET", str(output.write.call_args_list))
