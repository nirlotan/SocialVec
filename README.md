# SocialVec

SocialVec is a general framework of Social Embeddings for eliciting social world knowledge from social networks, which was developed by Nir Lotan and Dr. Einat Minkov as part of their research, available here: https://arxiv.org/abs/2111.03514

The repository provides several artifacts and demonstrations of SocialVec:

* Under the "models" folder you can find two traned models of SocialVec, applied on the social network of Twitter. The difference between these models is that one was implemented using CBOW, and the other using Skip Gram.
* Under the classification method you can find 7 twitter users attributes classifiers, which were developed based on SocialVec, and a usage example. These classifiers provides a binary classification of the user's age, gender, income and education levels, ethnisity (race) and having children or not, as detaied in the research.
* The main folder includes the implementation of a Streamlit application utilizing the SocialVec Social with some fun demo applications. This application can be accessed via: [https://share.streamlit.io/nirlotan/social2vec/app.py](https://nirlotan-socialvec-app-30uqjf.streamlitapp.com)

Feel free to contact us for any questions / requests for joint research based on this framework.

# Usage

## Prerequisites and imports
SocialVec implemetation is based on Gensim word2vec implementation. So the first thing you need to do is to import the gensim implementation of Word2Vec
```python
from gensim.models import Word2Vec
```
For additional details on Gensim python implementation see the Gensim website: https://radimrehurek.com/gensim/models/word2vec.html

## Load Model
Load the desired model - either the skip gram or CBOW version:
```python
# cbow_model 
SocialVec = Word2Vec.load("models/SocialVec_v3_350.model")
        
# skipgram_model 
SocialVec = Word2Vec.load("models/SocialVec_v6_sg_all.model")
```

Load the latest 2022 model directly from the web:
```python
SocialVec = pickle.load(urllib.request.urlopen("https://www.dropbox.com/s/qiuqdigicuxsavz/SocialVec2020_2022.pkl?dl=1"))
```

## User ID
The SocialVec model uses Twitter user IDs as the identifier of a user. 

For example, my Twitter username is @nirlotan, however if I want to check my similarity to other users, I need to use my user ID, which is: 40642926.

You can obtain the Twitter user ID of a user with 3rd party tools (some are available online), or by calling a simple [tweepy]https://github.com/tweepy/tweepy API call.

For additional details on tweepy usage, please refer to their documentation, however in brief, once the tweepy API is established, this is an easy way to get a userid given a username:

```python

api = tweepy.API(auth)
screen_name = "nirlotan"
user = api.get_user(screen_name)
ID = user.id_str
```

Once you have the relevant user IDs, you can use SocialVec for numerous insights, for example:

## Find similar users
Find the n most similar users. 
This query provides a list IDs of the users who are the most similar users to the requested ID based on the social similarity:
"topn" is the number of users requested
```python
similar_users = SocialVec.wv.most_similar([user_id], topn=10)
```

## Similarity Score
Get the social similarity score between two users (similarity score varies from -1 to 1):
```python
similarity = SocialVec.similarity(user1_id, user2_id)
```

## Analogy
As described in the paper, given 3 users, provide the 4th users. For example: @Android to @Google is like @Windows to ???
```python
user4_id = SocialVec.most_similar(negative=[user1_id], positive=[user2_id, user3_id], topn=1)
```
