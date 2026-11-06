# SocialVec

[![PyPI](https://img.shields.io/pypi/v/socialvec.svg)](https://pypi.org/project/socialvec/)
[![Python versions](https://img.shields.io/pypi/pyversions/socialvec.svg)](https://pypi.org/project/socialvec/)
[![Tests](https://github.com/nirlotan/SocialVec/actions/workflows/tests.yml/badge.svg)](https://github.com/nirlotan/SocialVec/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/nirlotan/SocialVec/blob/master/LICENSE)

**SocialVec** provides pre-trained *social embeddings* for about 200,000 popular Twitter accounts, learned from the follow graph of 1.3 million users. Accounts that are often followed by the same people end up close together, the same way words that appear in similar contexts do in word embeddings. The result captures *social world knowledge*: how people relate to entities, rather than facts about them.

SocialVec was developed by Nir Lotan and Einat Minkov. See the paper: [Social world knowledge: Modeling and applications](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0283700) (PLOS ONE, 2023).

With this package you can:

- get the 100-dimensional embedding of a popular account, by user ID or username;
- find similar accounts, for one account, a group of accounts, or any vector;
- represent any user as the average embedding of the accounts they follow;
- predict personal traits (political leaning, gender, age, and more) from an embedding;
- serve all of the above over a small HTTP API, or use it from the command line.

## Installation

```bash
pip install socialvec
```

The classifiers need TensorFlow and the HTTP API needs Flask. Both are optional extras:

```bash
pip install "socialvec[classifier]"        # adds TensorFlow
pip install "socialvec[api]"               # adds Flask and flask-cors
pip install "socialvec[classifier,api]"    # both
```

SocialVec supports Python 3.10 to 3.13.

## Quick start

```python
from socialvec.socialvec import SocialVec

sv = SocialVec()                    # downloads the model on first use, see "Models and data"
sv.get_similar("jack", topn=5)
```

```
  twitter_id  similarity   screen_name           name  description
0    6385432    0.841613         dickc   dick costolo  ...
1        989    0.831723            om             OM  ...
2    5746452    0.827466  waltmossberg  Walt Mossberg  ...
3   20536157    0.826462        Google         Google  ...
4    6708952    0.819312     SteveCase     Steve Case  ...
```

## Usage

### Embeddings

An account can be given as a user ID (an `int`, including numpy integers, or a string of digits) or as a username (case-insensitive):

```python
sv[12]                    # the same 100-dimensional vector, three ways
sv["12"]
sv["jack"]

sv.get_userid("NASA")       # '11348282'
sv.get_screen_name(813286)  # 'BarackObama'
```

### Similar accounts

`get_similar` takes a user ID, a username, a vector, or a list of IDs and usernames (whose embeddings are averaged). It returns a pandas DataFrame with the `topn` most similar accounts and their metadata.

```python
sv.get_similar(["Harvard", "MIT", "UCLA"], topn=5)
```

```
  twitter_id  similarity      screen_name                  name
0    5695032    0.867065             Yale       Yale University
1    5694822    0.861724        Princeton  Princeton University
2  248795646    0.850461         Columbia   Columbia University
3   14884486    0.845595  BrownUniversity      Brown University
4   33474655    0.840983    Cambridge_Uni  Cambridge University
```

```python
sv.get_similar(["FCBarcelona", "ManUtd", "realmadrid"], topn=5)
```

```
  twitter_id  similarity     screen_name              name
0  740336334    0.931517    GarethBale11       Gareth Bale
1  344801362    0.917337     DavidLuiz_4        David Luiz
2  140750163    0.915364       juanmata8  Juan Mata García
3  112764971    0.913976  FCBarcelona_es      FC Barcelona
4  533085085    0.912526             M10        Mesut Özil
```

### Similarity between two accounts

```python
sv.get_similarity("BarackObama", "HillaryClinton")    # 0.755
sv.get_similarity("BarackObama", "realDonaldTrump")   # 0.613
```

Either argument can also be a vector.

### Embedding any user

Most users aren't popular enough to have their own embedding. You can represent them by the average embedding of the accounts they follow. `get_average_embeddings` takes a list of user IDs and returns the average vector and the number of accounts it was based on. IDs that aren't in the model are skipped; if none are left, the vector is all zeros and the count is 0.

```python
following = [12, 989, 783214]           # e.g. the IDs a user follows, from the Twitter/X API
vector, count = sv.get_average_embeddings(following)
sv.get_similar(vector, topn=5)
```

### Classifiers

The classifiers predict personal traits from an embedding. They need the `classifier` extra (TensorFlow) and the `"2020c"` model they were trained on; calling `init_classifier()` on another model raises an error.

```python
sv = SocialVec(model_name="2020c")
sv.init_classifier()

sv.classifier.predict("political", sv["JoeBiden"])          # ('Democrat', 0.989)
sv.classifier.predict("political", sv["realDonaldTrump"])   # ('Republican', 0.866)
```

`predict` returns the predicted class and a confidence between 0 and 1, where 1 is high confidence and 0 is no confidence (which may be expected for accounts with no affiliation, e.g. a non-political account and the `political` classifier). To classify an ordinary user, pass the vector from `get_average_embeddings`.

| Classifier | Classes |
|---|---|
| `political` | Republican / Democrat |
| `gender` | Male / Female |
| `children` | Yes / No |
| `age` | Under25 / Over25 |
| `education` | Degree / HighSchool |
| `income` | High / Low |
| `ethnicity` | Caucasian / AfrAmerican |

`sv.classifier.classifier_names` lists them.

### Errors

Lookups raise `socialvec.exceptions.InvalidUserIdError` (a `ValueError`) for values that aren't a user ID or username, and `socialvec.exceptions.UserNotFoundError` (a `KeyError`) for accounts that aren't in the loaded model. An unknown `model_name` raises `UnknownModelError` (a `ValueError`), and a failed download raises `SocialVecError`. All of them subclass `socialvec.exceptions.SocialVecError`.

## Models and data

| `model_name` | Notes |
|---|---|
| `"2020"` | Default. |
| `"2020c"` | The model the classifiers were trained on. |
| `"2020_2022"` | |

On first use, SocialVec downloads the model (about 150 MB, or 300 MB for `"2020_2022"`) and the account metadata (about 30 MB). Later runs load them from disk.

- Files are kept in `~/.cache/socialvec`. To keep them somewhere else, set the `SOCIALVEC_HOME` environment variable to a folder.
- Files that versions up to 0.1.7.1 saved in the package folder or the working folder are still found and used.
- Every file is checked against a SHA-256 checksum before it is loaded.
- If the cache folder can't be written, the model is loaded into memory instead (and downloaded again next time).

## Command line

```bash
socialvec similar jack --topn 5
socialvec similarity BarackObama HillaryClinton
socialvec predict political JoeBiden          # uses the 2020c model; needs the classifier extra
socialvec download --model 2020c              # fetch a model ahead of time
```

`socialvec --help` lists all options. Every command takes `--model` to choose the model.

## HTTP API

SocialVec includes a small Flask API (install the `api` extra):

```bash
SOCIALVEC_MODEL=2020c python -m socialvec.api
```

It can also run under a WSGI server, e.g. `gunicorn "socialvec.api:create_app()"`. Importing `socialvec.api` does not load a model; `create_app()` does.

| Variable | Default | Meaning |
|---|---|---|
| `SOCIALVEC_MODEL` | `2020` | Model to load. `/predict` and `/predict_proba` need `2020c`; with other models they return 501. |
| `SOCIALVEC_CORS_ORIGINS` | `*` | Comma-separated origins allowed to call the API from a browser. |
| `SOCIALVEC_HOST` | `127.0.0.1` | Address to listen on. Use `0.0.0.0` to accept connections from other machines. |
| `PORT` | `8000` | Port to listen on. |

Endpoints:

- `GET /validate_userid?userid=`, `/validate_username?username=`, `/get_screen_name?userid=`, `/get_userid?username=`
- `GET /get_similar?input=&topn=` (`topn` is clamped to 1–100), `/get_embeddings?entity=`, `/get_similarity?entity1=&entity2=`
- `POST` with a JSON body: `/get_average_embeddings` `{"entity_list": [...]}`, `/predict` and `/predict_proba` `{"classifier_name": "political", "entity": "jack"}`

Errors are returned as `{"error": "..."}` with status 400 for bad input, 404 for accounts not in the model, and 500 (with no details) for unexpected errors, which are logged.

## Citation

If you use SocialVec in your research, please cite:

```bibtex
@article{lotan2023social,
  title   = {Social world knowledge: Modeling and applications},
  author  = {Lotan, Nir and Minkov, Einat},
  journal = {PLOS ONE},
  volume  = {18},
  number  = {7},
  pages   = {e0283700},
  year    = {2023},
  doi     = {10.1371/journal.pone.0283700}
}
```

## License

MIT. See [LICENSE](https://github.com/nirlotan/SocialVec/blob/master/LICENSE). The changelog is in [HISTORY.rst](https://github.com/nirlotan/SocialVec/blob/master/HISTORY.rst).
