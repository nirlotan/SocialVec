"""Tests for the `socialvec` command line, with the model mocked out."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from click.testing import CliRunner

from socialvec import __version__, cli
from socialvec.exceptions import UserNotFoundError


def _fake_sv():
    sv = MagicMock()
    sv.get_similar.return_value = pd.DataFrame({
        "twitter_id": ["6385432", "989"], "similarity": [0.84, 0.83],
        "screen_name": ["dickc", "om"], "name": ["dick costolo", "OM"], "description": ["", ""]})
    sv.get_similarity.return_value = np.float32(0.61277)
    sv.classifier.predict.return_value = ("Democrat", 0.98859)
    sv.__getitem__.return_value = np.zeros(100, dtype=np.float32)
    return sv


class TestCli(unittest.TestCase):

    def setUp(self):
        self.runner = CliRunner()
        self.sv = _fake_sv()
        patcher = patch("socialvec.cli._load", return_value=self.sv)
        self.load = patcher.start()
        self.addCleanup(patcher.stop)

    def test_help(self):
        result = self.runner.invoke(cli.main, ["--help"])
        self.assertEqual(0, result.exit_code)
        for command in ("similar", "similarity", "predict", "download"):
            self.assertIn(command, result.output)
        self.load.assert_not_called()

    def test_version(self):
        result = self.runner.invoke(cli.main, ["--version"])
        self.assertEqual(0, result.exit_code)
        self.assertIn(__version__, result.output)

    def test_similar(self):
        result = self.runner.invoke(cli.main, ["similar", "jack", "-n", "2"])
        self.assertEqual(0, result.exit_code, result.output)
        self.load.assert_called_once_with("2020")
        self.sv.get_similar.assert_called_once_with("jack", 2)
        self.assertIn("dickc", result.output)
        self.assertNotIn("description", result.output)

    def test_similar_model_option(self):
        self.runner.invoke(cli.main, ["similar", "jack", "--model", "2020_2022"])
        self.load.assert_called_once_with("2020_2022")

    def test_similar_bad_topn(self):
        result = self.runner.invoke(cli.main, ["similar", "jack", "-n", "0"])
        self.assertNotEqual(0, result.exit_code)
        self.load.assert_not_called()

    def test_similarity(self):
        result = self.runner.invoke(cli.main, ["similarity", "BarackObama", "realDonaldTrump"])
        self.assertEqual(0, result.exit_code, result.output)
        self.assertEqual("0.6128", result.output.strip())

    def test_predict_uses_classifier_model(self):
        result = self.runner.invoke(cli.main, ["predict", "political", "JoeBiden"])
        self.assertEqual(0, result.exit_code, result.output)
        self.load.assert_called_once_with("2020c")
        self.sv.init_classifier.assert_called_once()
        self.assertIn("Democrat", result.output)
        self.assertIn("0.989", result.output)

    def test_download(self):
        result = self.runner.invoke(cli.main, ["download", "--model", "2020c"])
        self.assertEqual(0, result.exit_code, result.output)
        self.load.assert_called_once_with("2020c")

    def test_run_shows_socialvec_errors_without_traceback(self):
        self.sv.get_similar.side_effect = UserNotFoundError("User 'nobody' not in SocialVec metadata")
        with patch("sys.argv", ["socialvec", "similar", "nobody"]):
            with patch("click.echo") as echo:
                code = cli.run()
        self.assertEqual(1, code)
        echo.assert_called_once()
        self.assertIn("nobody", echo.call_args.args[0])

    def test_run_success_returns_none(self):
        with patch("sys.argv", ["socialvec", "similarity", "a", "b"]):
            self.assertIsNone(cli.run())
