#!/usr/bin/env python

"""Tests for `socialvec` package."""


import unittest
from click.testing import CliRunner

from socialvec import socialvec as sv
from socialvec import cli


class TestSocialvec(unittest.TestCase):
    """Tests for `socialvec` package."""

    def setUp(self):
        """Set up test fixtures, if any."""
        self.SocialVec = sv.SocialVec()

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_validate_userid(self):

        uid = self.SocialVec.validate_userid(12)
        self.assertEqual('12', uid)

        #abc is an invalid user
        with self.assertRaises(Exception):
            self.SocialVec.validate_userid("abc")

        #11 is an invalid user
        with self.assertRaises(Exception):
            self.SocialVec.validate_userid("11")

    def test_validate_username(self):

        uid = self.SocialVec.validate_username("Twitter")
        self.assertEqual('783214', uid)

        #titter is an invalid user
        with self.assertRaises(Exception):
            self.SocialVec.validate_username("titter")


    def test_get_screen_name(self):
        name = self.SocialVec.get_screen_name(813286)
        self.assertEqual('BarackObama', name)

    def test_get_similar(self):
        """Test something."""
        sim = self.SocialVec.get_similar("415859364", 10)
        self.assertEqual("Nike.com", sim.iloc[0]['name'])

    def test_get_similar_text(self):
        """Test something."""
        sim = self.SocialVec.get_similar("Harvard", 10)
        self.assertEqual("Jordan", sim.iloc[0]['name'])

    def test_get_average_embeddings(self):
        v = self.SocialVec.get_average_embeddings([1, 12])
        self.assertEqual(v[1],1,msg="Error in expected length")

    def test_command_line_interface(self):
        """Test the CLI."""
        runner = CliRunner()
        result = runner.invoke(cli.main)
        assert result.exit_code == 0
        assert 'socialvec.cli.main' in result.output
        help_result = runner.invoke(cli.main, ['--help'])
        assert help_result.exit_code == 0
        assert '--help  Show this message and exit.' in help_result.output
