History
-------

0.1.0 (2022-09-29)
------------------

* First release on PyPI.

0.1.1 (2022-09-29)
------------------

* Include config.yaml in the distribution.

0.1.2 (2022-10-02)
------------------

* Rearrange config.yaml
* Support multiple versions of the SocialVec model
* Fix bug when searching for similarity using username

0.1.3 (2022-10-14)
------------------
* Initial version of SocialVecClassifier

0.1.4 (2022-11-08)
------------------
* Updates to SocialVecClassifier

0.1.5 (2023-11-07)
------------------
* Update a dedicated model for the SocialVecClassifier (2020c)

0.1.6 (2023-11-09)
------------------
* Modify requirements to support more up-to-date python versions

0.1.7 (2024-10-22)
------------------
* Add the option to load the model to RAM in case there is no write permission to the package folder (which

0.1.7.1 (2024-10-22)
--------------------
* Add pypi documentation

Unreleased
----------
* User IDs given as numpy integers (e.g. ``np.int64``) now work everywhere; ``get_average_embeddings`` no longer silently drops them.
* Errors are now ``socialvec.exceptions.InvalidUserIdError`` (a ``ValueError``) and ``UserNotFoundError`` (a ``KeyError``), both subclasses of ``SocialVecError``, instead of a bare ``Exception``.
* Looking up a username that is in the metadata but not in the loaded model raises ``UserNotFoundError`` instead of gensim's ``KeyError``.
* When several accounts share a screen name, the one that is in the loaded model is used.
* Username lookups are now a dictionary lookup instead of a scan of the whole metadata table.
* ``get_screen_name`` returns ``None`` for users that are in the model but have no metadata, instead of raising ``IndexError``.
* ``get_similar`` accepts a list of usernames as well as user IDs.
* ``init_classifier()`` raises a clear error for models with no classifiers (classifiers are only available for ``2020c``), and ``predict`` raises one for unknown classifier names.
* Fix ``f1_m`` (it referred to ``self`` from a static method).
* Remove the hard-coded vector size of 100.
* API (``socialvec.api``):

  * The model is chosen with ``SOCIALVEC_MODEL`` (default ``2020``). ``/predict`` and ``/predict_proba`` need ``2020c``, and return 501 with an explanation on other models instead of failing on every request.
  * Classifiers are loaded once when the app is created, not lazily on the first request.
  * App setup moved into ``create_app()``; importing the module no longer downloads or loads a model. ``python -m socialvec.api`` still works.
  * Listens on ``127.0.0.1`` by default instead of ``0.0.0.0``; set ``SOCIALVEC_HOST`` and ``PORT`` to change it. Allowed CORS origins come from ``SOCIALVEC_CORS_ORIGINS`` (default ``*``).
  * Errors: 400 for invalid input (including a non-integer ``topn`` and missing or non-JSON bodies), 404 for users not in the model, and a generic 500 for unexpected errors, which are logged instead of returned to the client.
  * ``/get_similar`` clamps ``topn`` to 1-100 and returns ``application/json``.
