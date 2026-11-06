"""Tests for the SocialVec HTTP API (`socialvec.api`).

These use a small in-memory stand-in for SocialVec, so they don't download
the models or need TensorFlow.
"""

import os
import subprocess
import sys
import types
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from socialvec import api
from socialvec.exceptions import InvalidUserIdError, SocialVecError, UserNotFoundError


USERS = {"12": "jack", "989": "om"}
VECTORS = {"12": np.array([1.0, 0.0], dtype=np.float32), "989": np.array([0.0, 1.0], dtype=np.float32)}


class FakeClassifier:
    classifier_names = ["political"]

    def _check(self, classifier_name):
        if classifier_name not in self.classifier_names:
            raise SocialVecError(f"Unknown classifier {classifier_name!r}")

    def predict(self, classifier_name, v):
        self._check(classifier_name)
        return ("Democrat", 0.75)

    def predict_proba(self, classifier_name, v):
        self._check(classifier_name)
        return 0.75


class FakeSocialVec:
    """Mimics the parts of SocialVec the API uses, including its exceptions."""

    def __init__(self, model_name="2020"):
        self.model_name = model_name
        self.init_classifier_calls = 0

    def init_classifier(self):
        self.init_classifier_calls += 1
        if self.model_name != "2020c":
            raise SocialVecError(f"No classifiers are available for model '{self.model_name}'.")
        self.classifier = FakeClassifier()

    def validate_userid(self, userid):
        if isinstance(userid, bool) or not (isinstance(userid, int) or (isinstance(userid, str) and userid.isdigit())):
            raise InvalidUserIdError(f"Invalid user id: {userid!r}")
        userid = str(userid)
        if userid not in USERS:
            raise UserNotFoundError(f"User {userid} not in SocialVec model {self.model_name}")
        return userid

    def get_userid(self, username):
        for userid, name in USERS.items():
            if name.lower() == username.lower():
                return userid
        raise UserNotFoundError(f"User '{username}' not in SocialVec metadata")

    validate_username = get_userid

    def get_screen_name(self, userid):
        return USERS[self.validate_userid(userid)]

    def _resolve(self, entity):
        if isinstance(entity, int) or (isinstance(entity, str) and entity.isdigit()):
            return self.validate_userid(entity)
        return self.get_userid(entity)

    def get_embeddings(self, entity):
        return VECTORS[self._resolve(entity)]

    def get_similar(self, input, topn=10):
        self.last_topn = topn
        userid = self._resolve(input)
        rows = [(other, 0.5) for other in USERS if other != userid][:topn]
        return pd.DataFrame(rows, columns=["twitter_id", "similarity"])

    def get_average_embeddings(self, entity_list):
        ids = []
        for entity in entity_list:
            try:
                ids.append(self.validate_userid(entity))
            except SocialVecError:
                continue
        if not ids:
            return np.zeros(2, dtype=np.float32), 0
        return np.mean([VECTORS[i] for i in ids], axis=0), len(ids)

    def get_similarity(self, entity1, entity2):
        return np.float32(np.dot(self.get_embeddings(entity1), self.get_embeddings(entity2)))


class ApiTestCase(unittest.TestCase):
    model_name = "2020"

    def setUp(self):
        self.sv = FakeSocialVec(self.model_name)
        self.app = api.create_app(sv=self.sv, cors_origins="*")
        self.client = self.app.test_client()


class TestLookups(ApiTestCase):

    def test_validate_userid(self):
        r = self.client.get("/validate_userid?userid=12")
        self.assertEqual(200, r.status_code)
        self.assertEqual({"userid": "12"}, r.get_json())

    def test_invalid_userid_is_400(self):
        r = self.client.get("/validate_userid?userid=abc")
        self.assertEqual(400, r.status_code)
        self.assertIn("Invalid user id", r.get_json()["error"])

    def test_unknown_userid_is_404(self):
        r = self.client.get("/validate_userid?userid=11")
        self.assertEqual(404, r.status_code)
        self.assertIn("not in SocialVec model", r.get_json()["error"])

    def test_unknown_username_is_404(self):
        r = self.client.get("/get_userid?username=nobody")
        self.assertEqual(404, r.status_code)

    def test_missing_parameter_is_400(self):
        for path in ("/validate_userid", "/validate_username", "/get_screen_name",
                     "/get_userid", "/get_similar", "/get_embeddings", "/get_similarity?entity1=12"):
            with self.subTest(path=path):
                r = self.client.get(path)
                self.assertEqual(400, r.status_code)
                self.assertIn("Missing required query parameter", r.get_json()["error"])

    def test_get_screen_name(self):
        r = self.client.get("/get_screen_name?userid=989")
        self.assertEqual({"screen_name": "om"}, r.get_json())

    def test_get_embeddings(self):
        r = self.client.get("/get_embeddings?entity=jack")
        self.assertEqual({"embeddings": [1.0, 0.0]}, r.get_json())

    def test_get_similarity(self):
        r = self.client.get("/get_similarity?entity1=12&entity2=jack")
        self.assertEqual({"similarity": 1.0}, r.get_json())

    def test_unknown_route_is_json_404(self):
        r = self.client.get("/no_such_endpoint")
        self.assertEqual(404, r.status_code)
        self.assertIn("error", r.get_json())


class TestGetSimilar(ApiTestCase):

    def test_default_topn(self):
        r = self.client.get("/get_similar?input=jack")
        self.assertEqual(200, r.status_code)
        self.assertEqual("application/json", r.mimetype)
        self.assertEqual([{"twitter_id": "989", "similarity": 0.5}], r.get_json())
        self.assertEqual(api.TOPN_DEFAULT, self.sv.last_topn)

    def test_non_integer_topn_is_400(self):
        for bad in ("abc", "1.5", "ten"):
            with self.subTest(topn=bad):
                r = self.client.get(f"/get_similar?input=jack&topn={bad}")
                self.assertEqual(400, r.status_code)
                self.assertIn("topn", r.get_json()["error"])

    def test_topn_is_clamped(self):
        for raw, expected in (("0", 1), ("-5", 1), ("5", 5), ("100", 100), ("100000", 100)):
            with self.subTest(topn=raw):
                r = self.client.get(f"/get_similar?input=jack&topn={raw}")
                self.assertEqual(200, r.status_code)
                self.assertEqual(expected, self.sv.last_topn)

    def test_unknown_input_is_404(self):
        r = self.client.get("/get_similar?input=nobody")
        self.assertEqual(404, r.status_code)


class TestJsonBodies(ApiTestCase):

    def test_average_embeddings(self):
        r = self.client.post("/get_average_embeddings", json={"entity_list": [12, "989", "11", "abc"]})
        self.assertEqual(200, r.status_code)
        self.assertEqual({"average_embeddings": [0.5, 0.5], "count": 2}, r.get_json())

    def test_average_embeddings_empty_list(self):
        r = self.client.post("/get_average_embeddings", json={"entity_list": []})
        self.assertEqual({"average_embeddings": [0.0, 0.0], "count": 0}, r.get_json())

    def test_non_json_bodies_are_400(self):
        for path in ("/get_average_embeddings", "/predict", "/predict_proba"):
            for kwargs in ({}, {"data": "not json", "content_type": "text/plain"},
                           {"data": "{broken", "content_type": "application/json"},
                           {"json": [1, 2, 3]}):
                with self.subTest(path=path, body=kwargs):
                    r = self.client.post(path, **kwargs)
                    self.assertEqual(400, r.status_code)
                    self.assertIn("error", r.get_json())

    def test_average_embeddings_requires_a_list(self):
        for body in ({}, {"entity_list": "12"}, {"entity_list": None}):
            with self.subTest(body=body):
                r = self.client.post("/get_average_embeddings", json=body)
                self.assertEqual(400, r.status_code)
                self.assertIn("entity_list", r.get_json()["error"])

    def test_predict_requires_fields(self):
        for body in ({}, {"entity": "jack"}, {"classifier_name": "political"},
                     {"classifier_name": 5, "entity": "jack"},
                     {"classifier_name": "political", "entity": True},
                     {"classifier_name": "political", "entity": ["jack"]}):
            for path in ("/predict", "/predict_proba"):
                with self.subTest(path=path, body=body):
                    r = self.client.post(path, json=body)
                    self.assertEqual(400, r.status_code)


class TestUnexpectedErrors(ApiTestCase):

    def test_unexpected_exception_is_generic_500(self):
        self.sv.get_userid = mock.Mock(side_effect=RuntimeError("secret internal detail"))
        with self.assertLogs("socialvec.api", level="ERROR"):
            r = self.client.get("/get_userid?username=jack")
        self.assertEqual(500, r.status_code)
        self.assertEqual({"error": "Internal server error"}, r.get_json())
        self.assertNotIn(b"secret", r.data)

    def test_other_socialvec_error_is_400(self):
        self.sv.get_userid = mock.Mock(side_effect=SocialVecError("bad config"))
        r = self.client.get("/get_userid?username=jack")
        self.assertEqual(400, r.status_code)
        self.assertEqual({"error": "bad config"}, r.get_json())


class TestClassifierWithoutClassifierModel(ApiTestCase):
    model_name = "2020"

    def test_predict_explains_which_model_is_needed(self):
        for path in ("/predict", "/predict_proba"):
            with self.subTest(path=path):
                r = self.client.post(path, json={"classifier_name": "political", "entity": "jack"})
                self.assertEqual(501, r.status_code)
                self.assertIn("SOCIALVEC_MODEL=2020c", r.get_json()["error"])

    def test_classifier_init_attempted_once_at_startup(self):
        self.assertEqual(1, self.sv.init_classifier_calls)
        self.client.post("/predict", json={"classifier_name": "political", "entity": "jack"})
        self.assertEqual(1, self.sv.init_classifier_calls)


class TestClassifierWith2020c(ApiTestCase):
    model_name = "2020c"

    def test_predict(self):
        r = self.client.post("/predict", json={"classifier_name": "political", "entity": "jack"})
        self.assertEqual(200, r.status_code)
        self.assertEqual({"affiliation": "Democrat", "confidence": 0.75}, r.get_json())

    def test_predict_with_numeric_id(self):
        r = self.client.post("/predict", json={"classifier_name": "political", "entity": 12})
        self.assertEqual(200, r.status_code)

    def test_predict_proba(self):
        r = self.client.post("/predict_proba", json={"classifier_name": "political", "entity": "12"})
        self.assertEqual(200, r.status_code)
        self.assertEqual({"probability": 0.75}, r.get_json())

    def test_unknown_classifier_is_400(self):
        r = self.client.post("/predict", json={"classifier_name": "nope", "entity": "jack"})
        self.assertEqual(400, r.status_code)
        self.assertIn("Unknown classifier", r.get_json()["error"])

    def test_unknown_entity_is_404(self):
        r = self.client.post("/predict", json={"classifier_name": "political", "entity": "nobody"})
        self.assertEqual(404, r.status_code)

    def test_classifier_initialized_once(self):
        for _ in range(3):
            self.client.post("/predict", json={"classifier_name": "political", "entity": "jack"})
        self.assertEqual(1, self.sv.init_classifier_calls)


class TestCreateApp(unittest.TestCase):

    def _fake_socialvec_module(self):
        module = types.ModuleType("socialvec.socialvec")
        module.SocialVec = FakeSocialVec
        return module

    def test_model_from_env(self):
        with mock.patch.dict(sys.modules, {"socialvec.socialvec": self._fake_socialvec_module()}), \
                mock.patch.dict(os.environ, {"SOCIALVEC_MODEL": "2020c"}):
            app = api.create_app()
        sv = app.extensions["socialvec"]["sv"]
        self.assertEqual("2020c", sv.model_name)
        r = app.test_client().post("/predict", json={"classifier_name": "political", "entity": "jack"})
        self.assertEqual(200, r.status_code)

    def test_default_model(self):
        env = {k: v for k, v in os.environ.items() if k != "SOCIALVEC_MODEL"}
        with mock.patch.dict(sys.modules, {"socialvec.socialvec": self._fake_socialvec_module()}), \
                mock.patch.dict(os.environ, env, clear=True):
            app = api.create_app()
        self.assertEqual("2020", app.extensions["socialvec"]["sv"].model_name)

    def test_cors_origins(self):
        app = api.create_app(sv=FakeSocialVec(), cors_origins="https://a.example, https://b.example")
        client = app.test_client()
        r = client.get("/validate_userid?userid=12", headers={"Origin": "https://a.example"})
        self.assertEqual("https://a.example", r.headers.get("Access-Control-Allow-Origin"))
        r = client.get("/validate_userid?userid=12", headers={"Origin": "https://evil.example"})
        self.assertIsNone(r.headers.get("Access-Control-Allow-Origin"))

    def test_cors_default_allows_any_origin(self):
        env = {k: v for k, v in os.environ.items() if k != "SOCIALVEC_CORS_ORIGINS"}
        with mock.patch.dict(os.environ, env, clear=True):
            app = api.create_app(sv=FakeSocialVec())
        r = app.test_client().get("/validate_userid?userid=12", headers={"Origin": "https://a.example"})
        # flask-cors echoes the request's origin rather than sending "*"
        self.assertIn(r.headers.get("Access-Control-Allow-Origin"), ("*", "https://a.example"))

    def test_main_reads_host_and_port_from_env(self):
        app = mock.Mock()
        with mock.patch.object(api, "create_app", return_value=app):
            with mock.patch.dict(os.environ, {"SOCIALVEC_HOST": "0.0.0.0", "PORT": "9000"}):
                api.main()
            app.run.assert_called_once_with(host="0.0.0.0", port=9000)

            app.run.reset_mock()
            env = {k: v for k, v in os.environ.items() if k not in ("SOCIALVEC_HOST", "PORT")}
            with mock.patch.dict(os.environ, env, clear=True):
                api.main()
            app.run.assert_called_once_with(host="127.0.0.1", port=8000)

    def test_import_does_not_load_model(self):
        code = ("import sys, socialvec.api; "
                "assert 'socialvec.socialvec' not in sys.modules, 'model module imported'")
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.assertEqual(0, result.returncode, result.stderr)


if __name__ == "__main__":
    unittest.main()
