"""Main module."""
import pandas as pd
import numpy as np
import pickle
import yaml
import os
import wget
from yaspin import yaspin
#from gensim.models import Word2Vec
import gzip
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow import keras

class SocialVec():
    def __init__(self, model_name="2020"):

        # Read configuration from config file
        current_folder = os.path.dirname(__file__)
        with open(os.path.join(current_folder, "config.yaml"), 'r') as f:
            self.config = yaml.load(f.read(), Loader=yaml.FullLoader)

        self.model_name = ""
        for m in self.config['models']:
            if m['name'] == model_name:
                model_filename = m['filename']
                model_path = m['download_url']
                self.model_name = model_name
        if self.model_name == "":
            raise Exception("model not found")

        if not os.path.exists(os.path.join(current_folder, model_filename)):
            # Load SocialVec model from the web
            print("First time model download")
            wget.download(model_path,
                          os.path.join(current_folder,model_filename))

        with yaspin(text="Initialize Model") as spinner:
            with gzip.open(os.path.join(current_folder, model_filename), 'rb') as pickle_file:
                self.sv = pickle.load(pickle_file)
                spinner.ok("✅ ")

        if not os.path.exists(os.path.join(current_folder, self.config['metadata'][0]['name'])):
            # Load SocialVec metadata from the web
            print("First time metadata download")
            wget.download(self.config['metadata'][1]['remote_path'],
                          os.path.join(current_folder,self.config['metadata'][0]['name']))

        with yaspin(text="Load Metadata") as spinner:
            self.entities = pd.read_parquet(os.path.join(current_folder, self.config['metadata'][0]['name']),
                                            engine="fastparquet")
            spinner.ok("✅ ")

    def validate_userid(self, userid) -> str:
        """
        validate the requested user. convert from int to string if needed

        Parameters
        ----------
        userid : twitter user id as it or string

        Returns
        -------
        user id in string format, or throw exception if user doesn't exist

        """
        if isinstance(userid, int):
            userid = str(userid)
        elif not userid.isdigit():
            raise Exception("User id must be an integer or a string with integer value")

        #check if user exists in the popular entities database
        if userid not in self.sv.wv.key_to_index.keys():
            raise Exception("User not in this SocialVec model version")

        return userid

    def validate_username(self, username) -> str:
        """
        validate the requested user. convert from int to string if needed

        Parameters
        ----------
        userid : twitter user id as it or string

        Returns
        -------
        user id in string format, or throw exception if user doesn't exist

        """

        user_row = self.entities[self.entities['screen_name'].str.lower() == username.lower()]
        #check if user exists in the popular entities database - CASE INSENSITIVE
        if user_row.shape[0] == 0:
            raise Exception("User not in SocialVec Metadata")

        #else:
        return user_row['twitter_id'].iloc[0]

    def get_screen_name(self, userid) -> str:
        """
        Get screen name for a given user ID
        Parameters
        ----------
        userid : string or int representing the twitter user ID

        Returns
        -------
        Twitter user name

        """
        userid = self.validate_userid(userid)
        return self.entities[self.entities['twitter_id'] == userid].iloc[0]['screen_name']


    def get_userid(self, username: str) -> str:
        """
        Get screen name for a given user ID
        Parameters
        ----------
        userid : string or int representing the twitter user ID

        Returns
        -------
        Twitter user name

        """
        return self.validate_username(username)

    def get_similar(self, input: str,topn: int = 10):
        """
        This function returns the topn similar entities for a given entity

        Parameters
        ----------
        input : twitter user id or username
        by : 'userid', 'username' or vector default is username
        topn : requested numner of similar entities

        Returns
        -------
        Pandas dataframe with the top n similar entities details

        """

        if isinstance(input, int) or (isinstance(input, str) and input.isdigit()):
            input = self.validate_userid(input)
        elif isinstance(input, str):
            input = self.validate_username(input)
        #else input is a 'vector'

        sim = self.sv.wv.most_similar(input, topn=topn)
        similar = pd.DataFrame(sim, columns=['twitter_id', 'similarity'])
        return pd.merge(similar, self.entities, on='twitter_id', how='left')

    def __getitem__(self, key):
        """
        Overload [] operator to return the entity embeddings

        Parameters
        ----------
        key - integer, string with user ID or string with username

        Returns
        -------
        SocialVec embeddings vector of the popular entity

        """
        if isinstance(key, int) or key.isdigit():
            userid = self.validate_userid(key)
        else:
            userid = self.validate_username(key)

        return self.sv.wv[userid]

    def get_embeddings(self, entity):
        """
        This function is a different way to trigger the [] operator
        """
        return self[entity]

    def get_average_embeddings(self, entity_list, type=""):
        """
        This function returns the average embeddings for a given list of Twitter user IDs

        Parameters
        ----------
        entity_list - list of entities IDs
        type - will be extended for future use with usernames

        Returns
        -------
        A tuple of the (1) average vector (2) the number of popular entities on which it was based

        """

        popular_entities = []
        for entity in entity_list:
            try:
                uid = self.validate_userid(entity)
                if uid in self.sv.wv.key_to_index.keys():
                    popular_entities.append(uid)
            except:
                continue

        if len(popular_entities) != 0:
            sv = np.mean(self.sv.wv[popular_entities], axis=0)
        else:
            sv = np.zeros(100)

        return sv, len(popular_entities)

    def get_similarity(self, entity1, entity2):
        """
        This function returns the similarity between two entities

        Parameters
        ----------
        entity1 - first entity ID
        entity2 - second entity ID

        Returns
        -------
        The cosine similarity score between the two entities

        """

        if type(entity1)==np.ndarray:
            v1 = entity1
        else:
            v1 = self[entity1]
        if type(entity2) == np.ndarray:
            v2 = entity2
        else:
            v2 = self[entity2]

        return cosine_similarity(v1.reshape(1, -1),v2.reshape(1, -1))[0][0]

    def init_classifier(self):
        self.classifier = self.SocialVecClassifier(self.model_name)

    class SocialVecClassifier():
        def __init__(self, model_name):

            # Read configuration from config file
            current_folder = os.path.dirname(__file__)
            with open(os.path.join(current_folder, "config.yaml"), 'r') as f:
                self.config = yaml.load(f.read(), Loader=yaml.FullLoader)

            self.model_name = ""
            for m in self.config['classification_models']:
                if m['name'] == model_name:
                    model_filename = m['filename']
                    self.model_name = model_name
            if self.model_name == "":
                raise Exception("model not found")        

            if not os.path.exists(os.path.join(current_folder, model_filename)):

                # Load SocialVec model from the web
                print("First time model download")
                wget.download(model_path,
                            os.path.join(current_folder,model_filename))

            self.political_model = keras.models.load_model(os.path.join(current_folder,model_filename))

        def predict_political_proba(self, v):
            """
            Return a number between 0 to 1 with the probabiliy in the selected class
            :param v:
            :type v:
            :return:
            :rtype:
            """
            return abs(self.political_model.predict(v.reshape(1, 100), verbose=False)[0][0] - 0.5)*2

        def predict_political(self, v):
            prediction = self.political_model.predict(v.reshape(1, 100), verbose=False)[0][0]

            # Original prediction is 0 for Democrat, 1 for Republican.
            # We convert it here to a confidence interval between 0 to 1 for either of the classes
            pred_proba = abs(prediction - 0.5)*2

            if round(prediction):
                affiliation =  'Republican'
            else:
                affiliation = 'Democrat'

            return (affiliation, pred_proba)

