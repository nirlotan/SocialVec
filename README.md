# SocialVec

SocialVec is a general framework of Social Embeddings for eliciting social world knowledge from social networks, which was developed by Nir Lotan and Dr. Einat Minkov as part of their research, available here: url to be added.

The repository provides several artifacts and demonstrations of SocialVec:

* Under the "models" folder you can find two traned models of SocialVec, applied on the social network of Twitter. The difference between these models is that one was implemented using CBOW, and the other using Skip Gram.
* Under the classification method you can find 7 twitter users attributes classifiers, which were developed based on SocialVec, and a usage example. These classifiers provides a binary classification of the user's age, gender, income and education levels, ethnisity (race) and having children or not, as detaied in the research.
* The main folder includes the implementation of a Streamlit application utilizing the SocialVec Social with some fun demo applications. This application can be accessed via: https://share.streamlit.io/nirlotan/social2vec/app.py

Feel free to contact us for any questions / requests for joint research based on this framework.

# Usage
SocialVec implemetation is based on Gensim word2vec implementation. For additional details on Gensim python implementation see the Gensim website: https://radimrehurek.com/gensim/models/word2vec.html

