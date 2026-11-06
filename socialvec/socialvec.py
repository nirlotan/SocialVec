"""Main module."""
import gzip
import numbers
import os
import pickle

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics.pairwise import cosine_similarity
from yaspin import yaspin

from ._download import fetch
from .exceptions import SocialVecError, InvalidUserIdError, UnknownModelError, UserNotFoundError

# TensorFlow is only needed by SocialVecClassifier, which imports it when it is created,
# so the embeddings can be used without TensorFlow installed.


def _read_config():
    with open(os.path.join(os.path.dirname(__file__), "config.yaml"), 'r') as f:
        return yaml.safe_load(f)


class SocialVec():
    def __init__(self, model_name="2020"):

        self.config = _read_config()

        entry = next((m for m in self.config['models'] if m['name'] == model_name), None)
        if entry is None:
            available = [m['name'] for m in self.config['models']]
            raise UnknownModelError(f"Unknown SocialVec model {model_name!r}; available models: {available}")
        self.model_name = model_name

        model_file = fetch(entry['filename'], entry['download_url'], entry.get('sha256'),
                           label=f"SocialVec {model_name} model")
        with yaspin(text="Initializing Model") as spinner:
            with gzip.open(model_file, 'rb') as pickle_file:
                self.sv = pickle.load(pickle_file)
            spinner.ok("✅ ")

        metadata = self.config['metadata']
        metadata_file = fetch(metadata['filename'], metadata['download_url'], metadata.get('sha256'),
                              label="SocialVec metadata")
        with yaspin(text="Load Metadata") as spinner:
            self.entities = pd.read_parquet(metadata_file, engine="fastparquet")
            spinner.ok("✅ ")

        self._build_lookup_tables()

    def _build_lookup_tables(self):
        """
        Build the username -> ID and ID -> screen name lookups once, at load time.

        A screen name can appear on more than one account (case-insensitively).
        Accounts that are in the loaded model are preferred, so a lookup by name
        finds the account that actually has an embedding.
        """
        ids = self.entities['twitter_id'].astype(str)
        names = self.entities['screen_name'].astype(str)
        in_model = ids.isin(self.sv.wv.key_to_index)

        names_df = pd.DataFrame({'name': names.str.lower(), 'twitter_id': ids, 'in_model': in_model})
        names_df = names_df.sort_values('in_model', ascending=False, kind='stable')
        names_df = names_df.drop_duplicates('name', keep='first')
        self._name_to_id = dict(zip(names_df['name'], names_df['twitter_id']))

        id_df = pd.DataFrame({'twitter_id': ids, 'screen_name': self.entities['screen_name']})
        id_df = id_df.drop_duplicates('twitter_id', keep='first')
        self._id_to_screen_name = dict(zip(id_df['twitter_id'], id_df['screen_name']))

    @staticmethod
    def _is_userid_like(value) -> bool:
        """True for ints (including numpy ints) and strings made only of digits."""
        if isinstance(value, bool):
            return False
        return isinstance(value, numbers.Integral) or (isinstance(value, str) and value.isdigit())

    @staticmethod
    def _normalize_userid(userid) -> str:
        """Convert an int / numpy int / digit string to the string ID used by the model."""
        if isinstance(userid, bool) or userid is None:
            raise InvalidUserIdError(f"Invalid user id: {userid!r}")
        if isinstance(userid, numbers.Integral):
            return str(int(userid))
        if isinstance(userid, str) and userid.isdigit():
            return userid
        raise InvalidUserIdError(
            f"User id must be an integer or a string with integer value, got {userid!r}")

    def _resolve(self, entity) -> str:
        """Resolve a user ID or a username to a user ID that is in the model."""
        if self._is_userid_like(entity):
            return self.validate_userid(entity)
        if isinstance(entity, str):
            return self.validate_username(entity)
        raise InvalidUserIdError(f"Expected a user ID or a username, got {entity!r}")

    def validate_userid(self, userid) -> str:
        """
        Validate the requested user ID and convert it to a string if needed.

        Parameters
        ----------
        userid : Twitter user ID, as an int (including numpy ints) or a string of digits

        Returns
        -------
        The user ID as a string.

        Raises
        ------
        InvalidUserIdError if the value is not a user ID,
        UserNotFoundError if the user is not in the loaded model.

        """
        userid = self._normalize_userid(userid)

        if userid not in self.sv.wv.key_to_index:
            raise UserNotFoundError(f"User {userid} not in SocialVec model {self.model_name}")

        return userid

    def validate_username(self, username) -> str:
        """
        Look up a username (case-insensitive) and return its user ID.

        Parameters
        ----------
        username : Twitter screen name, e.g. "jack"

        Returns
        -------
        The user ID (string) of an account with that name that is in the loaded model.

        Raises
        ------
        InvalidUserIdError if username is not a string,
        UserNotFoundError if no account with that name is in the loaded model.

        """
        userid = self.get_userid(username)
        if userid not in self.sv.wv.key_to_index:
            raise UserNotFoundError(
                f"User '{username}' is in the SocialVec metadata but not in model {self.model_name}")
        return userid

    def get_screen_name(self, userid):
        """
        Get the screen name for a given user ID.

        Parameters
        ----------
        userid : string or int representing the Twitter user ID

        Returns
        -------
        The Twitter screen name, or None if the user is in the model but has no metadata.

        Raises
        ------
        InvalidUserIdError / UserNotFoundError, as in validate_userid.

        """
        userid = self.validate_userid(userid)
        return self._id_to_screen_name.get(userid)

    def get_userid(self, username: str) -> str:
        """
        Get the user ID for a given screen name (case-insensitive).

        If several accounts share the name, an account that is in the loaded
        model is preferred. The returned account may still be one that only
        exists in the metadata; use validate_username to require an embedding.

        Parameters
        ----------
        username : Twitter screen name

        Returns
        -------
        The Twitter user ID as a string.

        Raises
        ------
        InvalidUserIdError if username is not a string,
        UserNotFoundError if the name is not in the SocialVec metadata.

        """
        if not isinstance(username, str):
            raise InvalidUserIdError(f"Username must be a string, got {username!r}")
        userid = self._name_to_id.get(username.lower())
        if userid is None:
            raise UserNotFoundError(f"User '{username}' not in SocialVec metadata")
        return userid

    def get_similar(self, input, topn: int = 10):
        """
        This function returns the topn similar entities for a given entity

        Parameters
        ----------
        input : a user ID, a username, an embedding vector, or a list of user IDs / usernames (whose embeddings are averaged)
        topn : requested number of similar entities

        Returns
        -------
        Pandas dataframe with the top n similar entities details

        """

        if isinstance(input, (list, tuple)):
            input = [item if isinstance(item, np.ndarray) else self._resolve(item) for item in input]
        elif not isinstance(input, np.ndarray):
            input = self._resolve(input)

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
        return self.sv.wv[self._resolve(key)]

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
        A tuple of the (1) average vector (2) the number of popular entities on which it was based.
        IDs that are not in the model are skipped; if none are left, the vector is all zeros.

        """

        popular_entities = []
        for entity in entity_list:
            try:
                popular_entities.append(self.validate_userid(entity))
            except SocialVecError:
                continue

        if len(popular_entities) != 0:
            sv = np.mean(self.sv.wv[popular_entities], axis=0)
        else:
            sv = np.zeros(self.sv.wv.vector_size, dtype=np.float32)

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

            current_folder = os.path.dirname(__file__)
            self.config = _read_config()

            # Classifiers are trained on one model's embedding space and can't be used with another
            entry = next((m for m in self.config['classification_models'] if m['name'] == model_name), None)
            if entry is None:
                available = [m['name'] for m in self.config['classification_models']]
                raise SocialVecError(
                    f"No classifiers are available for model '{model_name}'. "
                    f"Classifiers exist for: {available}, e.g. SocialVec(model_name='{available[0]}')")

            try:
                import tensorflow as tf
                from tensorflow import keras
            except ImportError as e:
                raise SocialVecError(
                    "The SocialVec classifiers need TensorFlow, which could not be imported "
                    f"({e}). Install it with: pip install 'socialvec[classifier]'") from e
            self._tf = tf

            keras.utils.get_custom_objects().update({'recall_m': self.recall_m,
                                                     'precision_m': self.precision_m,
                                                     'f1_m': self.f1_m})

            self.models_dict = {}
            self.models_classes = {}

            for classifier in entry['classifiers']:
                classifier_name = classifier['name']
                classifier_path = os.path.join(current_folder, classifier['filename'])
                if not os.path.exists(classifier_path):
                    raise SocialVecError(
                        f"Configuration error: classifier {classifier_name} not found at {classifier_path}")
                self.models_dict[classifier_name] = keras.models.load_model(classifier_path)
                self.models_classes[classifier_name] = {'class0': classifier['class0'],
                                                        'class1': classifier['class1']}

        @property
        def classifier_names(self):
            return list(self.models_dict)

        def _raw_predict(self, classifier_name, v):
            if classifier_name not in self.models_dict:
                raise SocialVecError(
                    f"Unknown classifier {classifier_name!r}; available: {self.classifier_names}")
            model = self.models_dict[classifier_name]
            param_to_predict = self._tf.convert_to_tensor(np.asarray(v, dtype=np.float32).reshape(1, -1))
            return model.predict(param_to_predict, verbose=False)[0][0]

        def predict_proba(self, classifier_name, v):
            """
            Return the confidence of the prediction, between 0 and 1.

            Despite the name, this is not the probability of a specific class:
            it is abs(p - 0.5) * 2 for the classifier's raw output p, i.e. 1 when the
            classifier is certain of either class and 0 when it can't tell them apart.
            Use predict() to also get which class was predicted.
            """
            return abs(self._raw_predict(classifier_name, v) - 0.5)*2

        def predict(self, classifier_name, v):
            """
            Classify an embedding vector.

            Returns
            -------
            A tuple (class label, confidence between 0 and 1); see predict_proba for the confidence.
            """
            prediction = self._raw_predict(classifier_name, v)

            # Original prediction is 0 for Democrat, 1 for Republican.
            # We convert it here to a confidence interval between 0 to 1 for either of the classes
            pred_proba = abs(prediction - 0.5)*2

            if round(prediction):
                affiliation =  self.models_classes[classifier_name]['class0']
            else:
                affiliation =  self.models_classes[classifier_name]['class1']

            return (affiliation, pred_proba)

        # Custom metrics the classifiers were saved with; Keras needs them to load the .h5 files.

        @staticmethod
        def recall_m(y_true, y_pred):
            from tensorflow.keras import backend as K
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall

        @staticmethod
        def precision_m(y_true, y_pred):
            from tensorflow.keras import backend as K
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            return precision

        @staticmethod
        def f1_m(y_true, y_pred):
            from tensorflow.keras import backend as K
            precision = SocialVec.SocialVecClassifier.precision_m(y_true, y_pred)
            recall = SocialVec.SocialVecClassifier.recall_m(y_true, y_pred)
            return 2*((precision*recall)/(precision+recall+K.epsilon()))
