#!/usr/bin/env python

"""Tests for `socialvec` package.

These tests download and load the real models on first run (~150 MB each),
so the models are loaded once per test class rather than once per test.
"""


import importlib.util
import unittest
from unittest.mock import patch

import numpy as np
from click.testing import CliRunner

from socialvec import socialvec as sv
from socialvec import cli
from socialvec.exceptions import InvalidUserIdError, SocialVecError, UserNotFoundError


HAS_TENSORFLOW = importlib.util.find_spec("tensorflow") is not None


class TestSocialvec(unittest.TestCase):
    """Tests for `socialvec` package, using the default model."""

    @classmethod
    def setUpClass(cls):
        cls.SocialVec = sv.SocialVec()

    def test_validate_userid(self):

        uid = self.SocialVec.validate_userid(12)
        self.assertEqual('12', uid)

        #abc is an invalid user
        with self.assertRaises(InvalidUserIdError):
            self.SocialVec.validate_userid("abc")

        #11 is an invalid user
        with self.assertRaises(UserNotFoundError):
            self.SocialVec.validate_userid("11")

    def test_validate_userid_numpy_int(self):
        self.assertEqual('12', self.SocialVec.validate_userid(np.int64(12)))

    def test_validate_userid_rejects_none_and_bool(self):
        for bad in (None, True, 1.5):
            with self.assertRaises(InvalidUserIdError):
                self.SocialVec.validate_userid(bad)

    def test_validate_username(self):

        uid = self.SocialVec.validate_username("Twitter")
        self.assertEqual('783214', uid)

        #titter is an invalid user
        with self.assertRaises(UserNotFoundError):
            self.SocialVec.validate_username("titter")

    def test_username_not_in_model(self):
        """A name that is only in the metadata raises the library's error, not gensim's."""
        model = self.SocialVec
        name = next(n for n, uid in model._name_to_id.items() if uid not in model.sv.wv.key_to_index)
        with self.assertRaises(UserNotFoundError):
            model[name]
        with self.assertRaises(UserNotFoundError):
            model.get_similar(name)

    def test_duplicate_username_prefers_account_in_model(self):
        """'DrShayPhD' is on two accounts; only 24333130 is in the 2020 model."""
        self.assertEqual('24333130', self.SocialVec.validate_username("DrShayPhD"))
        self.assertEqual(100, len(self.SocialVec["DrShayPhD"]))

    def test_get_screen_name(self):
        name = self.SocialVec.get_screen_name(813286)
        self.assertEqual('BarackObama', name)

    def test_get_screen_name_without_metadata(self):
        model = self.SocialVec
        uid = next(k for k in model.sv.wv.key_to_index if k not in model._id_to_screen_name)
        self.assertIsNone(model.get_screen_name(uid))

    def test_getitem(self):
        v = self.SocialVec[12]
        np.testing.assert_array_equal(v, self.SocialVec["12"])
        np.testing.assert_array_equal(v, self.SocialVec["jack"])
        np.testing.assert_array_equal(v, self.SocialVec[np.int64(12)])

    def test_get_similar(self):
        """Test something."""
        sim = self.SocialVec.get_similar("415859364", 10)
        self.assertEqual("Nike.com", sim.iloc[0]['name'])

    def test_get_similar_text(self):
        """The nearest neighbours of Harvard are other universities."""
        sim = self.SocialVec.get_similar("Harvard", 5)
        self.assertTrue({"MIT", "Stanford", "Yale", "Princeton"} & set(sim['screen_name']))

    def test_get_similar_list_of_names_and_ids(self):
        by_name = self.SocialVec.get_similar(["Harvard", "MIT", "UCLA"])
        ids = [self.SocialVec.get_userid(n) for n in ["Harvard", "MIT", "UCLA"]]
        by_id = self.SocialVec.get_similar(ids)
        self.assertEqual(list(by_name['twitter_id']), list(by_id['twitter_id']))

    def test_get_average_embeddings(self):
        v = self.SocialVec.get_average_embeddings([1, 12])
        self.assertEqual(v[1],1,msg="Error in expected length")

    def test_get_average_embeddings_numpy_ids(self):
        v, count = self.SocialVec.get_average_embeddings(list(np.array([12, 989])))
        self.assertEqual(2, count)

    def test_get_average_embeddings_empty(self):
        v, count = self.SocialVec.get_average_embeddings([1])
        self.assertEqual(0, count)
        self.assertEqual((self.SocialVec.sv.wv.vector_size,), v.shape)

    def test_init_classifier_unsupported_model(self):
        with self.assertRaises(SocialVecError):
            self.SocialVec.init_classifier()

    def test_command_line_similar(self):
        """The CLI against the real model; other CLI tests are in test_cli.py."""
        with patch("socialvec.cli._load", return_value=self.SocialVec):
            result = CliRunner().invoke(cli.main, ["similar", "Harvard", "-n", "3"])
        self.assertEqual(0, result.exit_code, result.output)
        self.assertIn("screen_name", result.output)


@unittest.skipUnless(HAS_TENSORFLOW, "tensorflow is not installed")
class TestSocialVecClassifier(unittest.TestCase):
    """Classifier tests, using the 2020c model (the one the classifiers were trained on)."""

    @classmethod
    def setUpClass(cls):
        cls.SocialVec = sv.SocialVec(model_name="2020c")
        cls.SocialVec.init_classifier()

    def test_predict(self):
        model = self.SocialVec
        for user, expected, classifier in [("BarackObama", "Democrat", "political"),
                                           ("ChickfilA", "Republican", "political"),
                                           ("RapSheet", "Male", "gender"),
                                           ("ChelseaHouska", "Female", "gender")]:
            label, confidence = model.classifier.predict(classifier, model[user])
            self.assertEqual(expected, label, msg=f"{classifier} / {user}")
            self.assertTrue(0 <= confidence <= 1)

    def test_unknown_classifier(self):
        with self.assertRaises(SocialVecError):
            self.SocialVec.classifier.predict("nope", self.SocialVec["jack"])

    def test_f1_m(self):
        f1 = sv.SocialVec.SocialVecClassifier.f1_m(np.array([1.0, 0.0]), np.array([1.0, 0.0]))
        self.assertAlmostEqual(1.0, float(f1), places=5)
