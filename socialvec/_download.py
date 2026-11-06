"""Find, download and verify the SocialVec data files (embedding models and metadata)."""
import hashlib
import io
import os
import warnings
from pathlib import Path

import requests
from yaspin import yaspin

from .exceptions import SocialVecError

PACKAGE_DIR = Path(__file__).resolve().parent

_CHUNK_SIZE = 1 << 20
# (connect, read) timeouts in seconds; the read timeout is per chunk, not for the whole file.
_TIMEOUT = (15, 120)


def cache_dir() -> Path:
    """The folder downloaded files are kept in: $SOCIALVEC_HOME, or ~/.cache/socialvec."""
    return Path(os.environ.get("SOCIALVEC_HOME") or Path.home() / ".cache" / "socialvec").expanduser()


def _existing_locations(filename):
    """Where the file may already be: the cache, then where versions up to 0.1.7.1 saved it."""
    locations = []
    for folder in (cache_dir(), PACKAGE_DIR, Path.cwd()):
        path = folder / filename
        if path not in locations:
            locations.append(path)
    return locations


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url, out):
    """Write the content at url to the binary file object out; return its sha256."""
    digest = hashlib.sha256()
    with requests.get(url, stream=True, timeout=_TIMEOUT) as response:
        response.raise_for_status()
        for chunk in response.iter_content(_CHUNK_SIZE):
            out.write(chunk)
            digest.update(chunk)
    return digest.hexdigest()


def _verify(actual, expected, label, url):
    if expected and actual != expected:
        raise SocialVecError(
            f"The {label} downloaded from {url} is corrupted or has changed: "
            f"expected sha256 {expected}, got {actual}.")


def _download_error(label, url, error):
    return SocialVecError(f"Could not download the {label} from {url}: {error}")


def fetch(filename, url, sha256=None, label=None):
    """
    Get a SocialVec data file, downloading it on first use.

    The file is looked up in the cache folder (see ``cache_dir``), then in the
    package folder and the current folder, where older versions saved it.
    If it isn't found (or its checksum doesn't match), it is downloaded to the
    cache folder. If the cache folder can't be written, the file is downloaded
    into memory instead.

    Parameters
    ----------
    filename : name of the file, e.g. "SocialVec2020.pkl.gz"
    url : where to download it from
    sha256 : expected sha256 hex digest; the file is not used if it doesn't match
    label : what the file is, for messages

    Returns
    -------
    A ``pathlib.Path`` to the file, or an ``io.BytesIO`` with its contents.

    Raises
    ------
    SocialVecError if the file can't be downloaded or doesn't match the checksum.

    """
    label = label or filename

    for path in _existing_locations(filename):
        if not path.is_file():
            continue
        if not sha256 or _file_sha256(path) == sha256:
            return path
        warnings.warn(f"Ignoring {path}: it does not match the expected checksum of the {label}.")

    target = cache_dir() / filename
    partial = target.with_name(target.name + ".part")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with yaspin(text=f"Downloading the {label} to {target}") as spinner:
            try:
                with open(partial, "wb") as f:
                    actual = _download(url, f)
                _verify(actual, sha256, label, url)
                os.replace(partial, target)
            except BaseException:
                spinner.fail("❌ ")
                try:
                    partial.unlink()
                except OSError:
                    pass
                raise
            spinner.ok("✅ ")
        return target
    except requests.RequestException as e:
        # RequestException is an OSError too, so this must come before the OSError handler.
        raise _download_error(label, url, e) from e
    except OSError as e:
        warnings.warn(
            f"Could not save the {label} to {target.parent} ({e}); loading it into memory instead, "
            f"so it will be downloaded again next time. Set SOCIALVEC_HOME to a writable folder to avoid this.")

    buffer = io.BytesIO()
    with yaspin(text=f"Downloading the {label} into memory") as spinner:
        try:
            actual = _download(url, buffer)
            _verify(actual, sha256, label, url)
        except requests.RequestException as e:
            spinner.fail("❌ ")
            raise _download_error(label, url, e) from e
        except BaseException:
            spinner.fail("❌ ")
            raise
        spinner.ok("✅ ")
    buffer.seek(0)
    return buffer
