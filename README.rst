=========
SocialVec
=========

.. image:: https://img.shields.io/pypi/v/socialvec.svg
   :target: https://pypi.python.org/pypi/socialvec

The **SocialVec** package provides pre-trained embeddings for approximately 200,000 popular Twitter accounts. **SocialVec** is a framework for learning social entity embeddings, derived from a large-scale Twitter dataset encompassing 1.3 million users and the accounts they follow.

* Free software: MIT license

What are SocialVec Embeddings?
==============================

**SocialVec embeddings** are low-dimensional vector representations of popular Twitter accounts. These embeddings are trained on co-occurrence patterns observed in the Twitter social network. Accounts frequently co-followed by users are considered socially related, making these embeddings similar to word embeddings where words in similar contexts have similar vector representations.

Package Features
================

This package includes the following features:

- **Access to pre-trained SocialVec embeddings:**

  - Pre-trained embeddings for approximately 200,000 popular Twitter accounts.
  - Embeddings are 100-dimensional, trained using the Skip-gram model with negative sampling (SGNS).

- **Entity similarity computation:**

  - Calculate cosine similarity between SocialVec embeddings to assess social similarity between entities.
  - Enables tasks like:

    - Identifying similar entities (e.g., universities similar to UC Berkeley).
    - Recommending Twitter accounts based on existing followings.
    - Assessing the political leaning of news sources.

- **Entity analogy exploration:**

  - Experiment with relational arithmetic on SocialVec embeddings to explore entity analogies, similar to word analogies.

Potential Applications
======================

The **SocialVec** package can be used for a wide range of tasks, including:

- **Recommendation systems:** Recommending Twitter accounts or other content based on user social affinity captured by the embeddings.
- **Social analysis:** Investigating social trends and relationships between entities on Twitter.
- **Bias detection:** Identifying potential biases in social media content or user behavior based on social context.
- **Inferring personal traits:** Predicting user characteristics like age, gender, or political leaning based on their social connections on Twitter.

Examples
========

Here are some practical examples of what you can do with **SocialVec**:

- **Finding similar entities:** Retrieve universities similar to UC Berkeley based on the cosine similarity of their SocialVec embeddings.
- **Recommending Twitter accounts:** Suggest accounts similar to those followed by a specific user, leveraging social context captured in the embeddings.
- **Assessing political leaning:** Determine the political bias of news sources by comparing their similarity to embeddings of politically polarized accounts (e.g., accounts of prominent politicians).
- **Exploring entity analogies:** Complete analogies like *"X-Factor : Simon Cowell :: The Voice : ?"* using vector arithmetic on SocialVec embeddings.

Advantages of SocialVec
=======================

- **Captures social world knowledge:** Unlike embeddings derived from factual knowledge bases like Wikipedia or Wikidata, SocialVec embeddings reflect relationships between entities based on social media interactions.
- **Wider coverage:** SocialVec represents a broader range of entities, as many Twitter accounts do not have corresponding Wikipedia pages.

Notes
=====

This README covers the pre-trained embeddings provided by the package. Specific implementation details and additional functionality will be defined as part of the package's development.

Credits
=======

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
