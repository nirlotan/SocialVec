.. highlight:: shell

============
Installation
============


Stable release
--------------

To install SocialVec, run this command in your terminal:

.. code-block:: console

    $ pip install socialvec

This is the preferred method to install SocialVec, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

The classifiers need TensorFlow and the API server needs Flask; install them with the optional extras:

.. code-block:: console

    $ pip install "socialvec[classifier]"
    $ pip install "socialvec[api]"

The models and metadata are downloaded on first use to ``~/.cache/socialvec``.
Set the ``SOCIALVEC_HOME`` environment variable to keep them in another folder.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The sources for SocialVec can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone https://github.com/nirlotan/SocialVec.git

Or download the `tarball`_:

.. code-block:: console

    $ curl -OJL https://github.com/nirlotan/SocialVec/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ pip install .

For development, install it in editable mode with the development tools:

.. code-block:: console

    $ pip install -r requirements_dev.txt -e ".[classifier,api]"
    $ pytest


.. _Github repo: https://github.com/nirlotan/SocialVec
.. _tarball: https://github.com/nirlotan/SocialVec/tarball/master
