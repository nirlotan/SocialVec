"""Tests for choosing, finding and downloading the model files, with the network mocked."""

import gzip
import hashlib
import io
import os
import pickle
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import requests
from gensim.models import KeyedVectors

from socialvec import _download
from socialvec import socialvec as sv
from socialvec.exceptions import SocialVecError, UnknownModelError

USER_IDS = ["12", "989"]


def _model_bytes(value):
    """A gzipped pickle shaped like the real models: an object with a .wv KeyedVectors."""
    kv = KeyedVectors(vector_size=4)
    kv.add_vectors(USER_IDS, np.full((len(USER_IDS), 4), value, dtype=np.float32))
    return gzip.compress(pickle.dumps(SimpleNamespace(wv=kv)))


def _metadata_bytes(tmp_path):
    path = tmp_path / "metadata-source.parquet"
    pd.DataFrame({"twitter_id": USER_IDS, "screen_name": ["jack", "om"]}).to_parquet(
        path, engine="fastparquet", compression="GZIP")
    return path.read_bytes()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


class FakeServer:
    """Stands in for requests.get, serving fixed bytes per URL and recording requests."""

    def __init__(self, files):
        self.files = files
        self.requested = []

    def get(self, url, stream=False, timeout=None):
        self.requested.append(url)
        response = mock.MagicMock()
        response.__enter__.return_value = response
        if url in self.files:
            data = self.files[url]
            response.iter_content.side_effect = lambda size: (data[i:i + size] for i in range(0, len(data), size))
        else:
            response.raise_for_status.side_effect = requests.HTTPError(f"404 Client Error for url: {url}")
        return response


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Isolated cache, package and working folders, a fake config and a fake server."""
    files = {
        "https://example.test/first.pkl.gz": _model_bytes(1.0),
        "https://example.test/middle.pkl.gz": _model_bytes(2.0),
        "https://example.test/last.pkl.gz": _model_bytes(3.0),
        "https://example.test/metadata.parquet.gz": _metadata_bytes(tmp_path),
    }
    config = {
        "models": [
            {"name": name, "filename": f"{name}.pkl.gz",
             "download_url": f"https://example.test/{name}.pkl.gz",
             "sha256": _sha(files[f"https://example.test/{name}.pkl.gz"])}
            for name in ("first", "middle", "last")
        ],
        "metadata": {"filename": "metadata.parquet.gz",
                     "download_url": "https://example.test/metadata.parquet.gz",
                     "sha256": _sha(files["https://example.test/metadata.parquet.gz"])},
        "classification_models": [],
    }
    for name in ("cache", "pkg", "cwd"):
        (tmp_path / name).mkdir()
    monkeypatch.setenv("SOCIALVEC_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(_download, "PACKAGE_DIR", tmp_path / "pkg")
    monkeypatch.chdir(tmp_path / "cwd")
    monkeypatch.setattr(sv, "_read_config", lambda: config)
    server = FakeServer(files)
    monkeypatch.setattr(_download.requests, "get", server.get)
    return SimpleNamespace(tmp=tmp_path, cache=tmp_path / "cache", config=config, files=files, server=server)


def _value(model):
    return float(model.sv.wv["12"][0])


def test_cache_dir_default(monkeypatch):
    monkeypatch.delenv("SOCIALVEC_HOME", raising=False)
    assert _download.cache_dir() == _download.Path.home() / ".cache" / "socialvec"


def test_unknown_model_lists_available_names(env):
    with pytest.raises(UnknownModelError) as info:
        sv.SocialVec(model_name="nope")
    assert isinstance(info.value, ValueError)
    assert "'first', 'middle', 'last'" in str(info.value)
    assert env.server.requested == []


@pytest.mark.parametrize("name, value", [("first", 1.0), ("middle", 2.0), ("last", 3.0)])
def test_downloads_selected_model_to_cache(env, name, value):
    model = sv.SocialVec(model_name=name)
    assert _value(model) == value
    assert env.server.requested == [f"https://example.test/{name}.pkl.gz",
                                     "https://example.test/metadata.parquet.gz"]
    assert {p.name for p in env.cache.iterdir()} == {"metadata.parquet.gz", f"{name}.pkl.gz"}
    assert model.get_screen_name(12) == "jack"

    # Second load comes from the cache.
    env.server.requested.clear()
    assert _value(sv.SocialVec(model_name=name)) == value
    assert env.server.requested == []


@pytest.mark.parametrize("name, value", [("first", 1.0), ("middle", 2.0)])
def test_unwritable_cache_loads_selected_model_into_memory(env, monkeypatch, name, value):
    """Regression: the in-memory fallback used to load the last model in the config."""
    blocker = env.tmp / "not-a-folder"
    blocker.write_text("")
    monkeypatch.setenv("SOCIALVEC_HOME", str(blocker / "cache"))

    with pytest.warns(UserWarning, match="into memory"):
        model = sv.SocialVec(model_name=name)

    assert _value(model) == value
    assert env.server.requested == [f"https://example.test/{name}.pkl.gz",
                                    "https://example.test/metadata.parquet.gz"]
    assert model.get_screen_name(12) == "jack"


def test_files_from_older_versions_are_used(env):
    """Files saved by older versions in the package folder or the working folder aren't downloaded again."""
    (env.tmp / "pkg" / "middle.pkl.gz").write_bytes(env.files["https://example.test/middle.pkl.gz"])
    (env.tmp / "cwd" / "metadata.parquet.gz").write_bytes(env.files["https://example.test/metadata.parquet.gz"])

    assert _value(sv.SocialVec(model_name="middle")) == 2.0
    assert env.server.requested == []
    assert list(env.cache.iterdir()) == []


def test_corrupted_cached_file_is_downloaded_again(env):
    (env.cache / "first.pkl.gz").write_bytes(b"truncated")
    with pytest.warns(UserWarning, match="checksum"):
        model = sv.SocialVec(model_name="first")
    assert _value(model) == 1.0
    assert (env.cache / "first.pkl.gz").read_bytes() == env.files["https://example.test/first.pkl.gz"]


def test_network_error_raises_socialvec_error(env, monkeypatch):
    def offline(url, **kwargs):
        raise requests.ConnectionError("Name or service not known")
    monkeypatch.setattr(_download.requests, "get", offline)

    with pytest.raises(SocialVecError, match="Could not download the SocialVec first model"):
        sv.SocialVec(model_name="first")
    assert list(env.cache.iterdir()) == []


def test_network_error_in_memory_fallback_raises_socialvec_error(env, monkeypatch):
    blocker = env.tmp / "not-a-folder"
    blocker.write_text("")
    monkeypatch.setenv("SOCIALVEC_HOME", str(blocker / "cache"))
    del env.files["https://example.test/first.pkl.gz"]

    with pytest.warns(UserWarning, match="into memory"), pytest.raises(SocialVecError, match="404"):
        sv.SocialVec(model_name="first")


def test_http_error_raises_socialvec_error(env):
    del env.files["https://example.test/last.pkl.gz"]
    with pytest.raises(SocialVecError, match="404"):
        sv.SocialVec(model_name="last")
    assert list(env.cache.iterdir()) == []


def test_checksum_mismatch_raises_and_keeps_nothing(env):
    env.files["https://example.test/first.pkl.gz"] = b"<html>Dropbox error page</html>"
    with pytest.raises(SocialVecError, match="corrupted or has changed"):
        sv.SocialVec(model_name="first")
    assert list(env.cache.iterdir()) == []


def test_fetch_returns_buffer_in_memory(env, monkeypatch):
    monkeypatch.setattr(_download, "cache_dir", lambda: env.tmp / "missing" / "x")
    (env.tmp / "missing").write_text("")
    data = env.files["https://example.test/first.pkl.gz"]
    with pytest.warns(UserWarning):
        result = _download.fetch("first.pkl.gz", "https://example.test/first.pkl.gz", _sha(data))
    assert isinstance(result, io.BytesIO)
    assert result.read() == data
