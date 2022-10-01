"""Main module."""
import pandas as pd
import numpy as np
import pickle
import yaml
import os
import wget
from yaspin import yaspin
from gensim.models import Word2Vec
import gzip


class SocialVec():
    def __init__(self):

        # Read configuration from config file
        current_folder = os.path.dirname(__file__)
        with open(os.path.join(current_folder, "config.yaml"), 'r') as f:
            self.config = yaml.load(f.read(), Loader=yaml.FullLoader)

        if not os.path.exists(os.path.join(current_folder, self.config["local_model"])):
            # Load SocialVec model from the web
            print("First time model download")
            wget.download(self.config["default_model_url"],
                          os.path.join(current_folder,self.config["local_model"]))

        with yaspin(text="Initialize Model") as spinner:
            with gzip.open(os.path.join(current_folder, self.config["local_model"]), 'rb') as pickle_file:
                self.sv = pickle.load(pickle_file)
                spinner.ok("✅ ")

        if not os.path.exists(os.path.join(current_folder, self.config["local_metadata"])):
            # Load SocialVec metadata from the web
            print("First time metadata download")
            wget.download(self.config["default_metadata_url"],
                          os.path.join(current_folder,self.config["local_metadata"]))

        with yaspin(text="Load Metadata") as spinner:
            self.entities = pd.read_parquet(os.path.join(current_folder, self.config["local_metadata"]),
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
        if self.entities[self.entities['twitter_id'] == userid].shape[0] == 0:
            raise Exception("User not in SocialVec Metadata")

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
            userid = self.validate_username(input)
        #else input is a 'vector'

        sim = self.sv.wv.most_similar(input, topn=topn)
        similar = pd.DataFrame(sim, columns=['twitter_id', 'similarity'])
        return pd.merge(similar, self.entities, on='twitter_id', how='left')

    def __getitem__(self, key):
        if isinstance(key, int) or key.isdigit():
            userid = self.validate_userid(key)
        else:
            userid = self.validate_username(key)

        return self.sv.wv[userid]

    def get_embeddings(self, entity):
        return self[entity]

    def get_average_embeddings(self, entity_list, type=""):

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

