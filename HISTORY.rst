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
* Downloaded models and metadata are now kept in ``~/.cache/socialvec`` (or the folder in the ``SOCIALVEC_HOME`` environment variable) instead of the package folder. Files that older versions saved in the package folder or the working folder are still used, so they are not downloaded again.
* Downloads are checked against a SHA-256 checksum before they are loaded.
* A failed download now raises ``SocialVecError`` instead of leaving a broken ``SocialVec`` object that fails later with ``AttributeError``.
* Fix loading the wrong model (always ``2020_2022``) when the package folder was not writable.
* The metadata is downloaded once and cached, instead of being read from Dropbox every time when the package folder was not writable.
* An unknown ``model_name`` raises ``UnknownModelError`` (a ``ValueError``) listing the available models.
* TensorFlow is now an optional extra: ``pip install "socialvec[classifier]"``. The embeddings work without it; ``init_classifier()`` raises a clear error if it is missing. Flask is an optional extra too: ``pip install "socialvec[api]"``.
* ``wget`` is no longer a dependency; downloads use ``requests``.
* Packaging: ship ``config.yaml`` and the classifier ``.h5`` files in the package, and never the downloaded models.
