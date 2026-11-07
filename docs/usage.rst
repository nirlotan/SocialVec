=====
Usage
=====

To use SocialVec in a project:

.. code-block:: python

    from socialvec.socialvec import SocialVec

    sv = SocialVec()                      # the default "2020" model
    sv.get_similar("jack", topn=5)        # most similar accounts, as a pandas DataFrame
    sv.get_similarity("BarackObama", "HillaryClinton")
    vector, count = sv.get_average_embeddings([12, 989, 783214])

The classifiers need the ``classifier`` extra and the ``"2020c"`` model:

.. code-block:: python

    sv = SocialVec(model_name="2020c")
    sv.init_classifier()
    sv.classifier.predict("political", sv["JoeBiden"])   # ('Democrat', 0.989)

See the :doc:`readme` for every feature, the command line and the HTTP API,
and :doc:`modules` for the API reference.
