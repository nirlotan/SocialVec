"""HTTP API for SocialVec.

Run it with ``python -m socialvec.api``, or with a WSGI server, e.g.
``gunicorn "socialvec.api:create_app()"``.

Configuration (environment variables):

* ``SOCIALVEC_MODEL`` - the model to load (default ``"2020"``). The classifier
  endpoints (``/predict``, ``/predict_proba``) need ``"2020c"``.
* ``SOCIALVEC_CORS_ORIGINS`` - comma-separated list of origins allowed to call
  the API from a browser (default ``*``, any origin).
* ``SOCIALVEC_HOST`` - address to listen on (default ``127.0.0.1``; use
  ``0.0.0.0`` to accept connections from other machines).
* ``PORT`` - port to listen on (default ``8000``).

``SOCIALVEC_HOST`` and ``PORT`` only apply to ``python -m socialvec.api``.
"""
import logging
import os

import numpy as np
from flask import Blueprint, Flask, current_app, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

from socialvec.exceptions import InvalidUserIdError, SocialVecError, UserNotFoundError

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "2020"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_CORS_ORIGINS = "*"

TOPN_DEFAULT = 10
TOPN_MIN = 1
TOPN_MAX = 100

_EXTENSION_KEY = "socialvec"


class ApiError(Exception):
    """An error caused by the request (missing or malformed input) or by the server setup."""

    def __init__(self, message, status=400):
        super().__init__(message)
        self.message = message
        self.status = status


bp = Blueprint("socialvec", __name__)


def _state():
    return current_app.extensions[_EXTENSION_KEY]


def _sv():
    return _state()["sv"]


def _classifier():
    state = _state()
    if state["classifier_error"] is not None:
        raise ApiError(state["classifier_error"], status=501)
    return state["sv"].classifier


def _require_arg(name):
    value = request.args.get(name)
    if value is None or value == "":
        raise ApiError(f"Missing required query parameter '{name}'")
    return value


def _json_body():
    data = request.get_json(silent=True)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ApiError("Request body must be a JSON object")
    return data


def _require_entity(data, name):
    value = data.get(name)
    if value is None or value == "" or isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ApiError(f"'{name}' is required and must be a user ID or a username")
    return value


def _require_classifier_name(data):
    value = data.get("classifier_name")
    if not isinstance(value, str) or not value:
        raise ApiError("'classifier_name' is required and must be a string")
    return value


def _parse_topn():
    raw = request.args.get("topn")
    if raw is None or raw == "":
        return TOPN_DEFAULT
    try:
        topn = int(raw)
    except ValueError:
        raise ApiError(f"'topn' must be an integer, got {raw!r}")
    return max(TOPN_MIN, min(TOPN_MAX, topn))


@bp.route('/validate_userid', methods=['GET'])
def validate_userid():
    result = _sv().validate_userid(_require_arg('userid'))
    return jsonify({'userid': result})


@bp.route('/validate_username', methods=['GET'])
def validate_username():
    result = _sv().validate_username(_require_arg('username'))
    return jsonify({'userid': result})


@bp.route('/get_screen_name', methods=['GET'])
def get_screen_name():
    result = _sv().get_screen_name(_require_arg('userid'))
    return jsonify({'screen_name': result})


@bp.route('/get_userid', methods=['GET'])
def get_userid():
    result = _sv().get_userid(_require_arg('username'))
    return jsonify({'userid': result})


@bp.route('/get_similar', methods=['GET'])
def get_similar():
    input_val = _require_arg('input')
    topn = _parse_topn()
    df = _sv().get_similar(input_val, topn)
    return current_app.response_class(df.to_json(orient='records'), mimetype='application/json')


@bp.route('/get_embeddings', methods=['GET'])
def get_embeddings():
    emb = _sv().get_embeddings(_require_arg('entity'))
    return jsonify({'embeddings': np.asarray(emb).tolist()})


@bp.route('/get_average_embeddings', methods=['POST'])
def get_average_embeddings():
    data = _json_body()
    entity_list = data.get('entity_list')
    if not isinstance(entity_list, list):
        raise ApiError("'entity_list' is required and must be a list of user IDs")
    avg_emb, count = _sv().get_average_embeddings(entity_list)
    return jsonify({'average_embeddings': np.asarray(avg_emb).tolist(), 'count': int(count)})


@bp.route('/get_similarity', methods=['GET'])
def get_similarity():
    sim = _sv().get_similarity(_require_arg('entity1'), _require_arg('entity2'))
    return jsonify({'similarity': float(sim)})


@bp.route('/predict', methods=['POST'])
def predict():
    data = _json_body()
    classifier_name = _require_classifier_name(data)
    entity = _require_entity(data, 'entity')
    classifier = _classifier()
    emb = _sv().get_embeddings(entity)
    affiliation, pred_proba = classifier.predict(classifier_name, np.asarray(emb))
    return jsonify({'affiliation': affiliation, 'confidence': float(pred_proba)})


@bp.route('/predict_proba', methods=['POST'])
def predict_proba():
    data = _json_body()
    classifier_name = _require_classifier_name(data)
    entity = _require_entity(data, 'entity')
    classifier = _classifier()
    emb = _sv().get_embeddings(entity)
    proba = classifier.predict_proba(classifier_name, np.asarray(emb))
    return jsonify({'probability': float(proba)})


def _error(message, status):
    return jsonify({'error': message}), status


@bp.app_errorhandler(ApiError)
def _handle_api_error(e):
    return _error(e.message, e.status)


@bp.app_errorhandler(InvalidUserIdError)
def _handle_invalid_userid(e):
    return _error(str(e), 400)


@bp.app_errorhandler(UserNotFoundError)
def _handle_user_not_found(e):
    return _error(str(e), 404)


@bp.app_errorhandler(SocialVecError)
def _handle_socialvec_error(e):
    return _error(str(e), 400)


@bp.app_errorhandler(HTTPException)
def _handle_http_exception(e):
    return _error(e.description, e.code)


@bp.app_errorhandler(Exception)
def _handle_unexpected(e):
    logger.exception("Unhandled error in %s %s", request.method, request.path)
    return _error("Internal server error", 500)


def _init_classifier(sv):
    """Load the classifiers once, up front. Returns an error message if they aren't available."""
    try:
        sv.init_classifier()
    except SocialVecError as e:
        logger.info("Classifier endpoints are disabled: %s", e)
        return (f"Classifiers are not available with model '{sv.model_name}': {e} "
                f"Start the API with SOCIALVEC_MODEL=2020c to enable them.")
    return None


def _cors_origins(value):
    origins = [o.strip() for o in value.split(",") if o.strip()]
    if not origins or origins == ["*"]:
        return "*"
    return origins


def create_app(sv=None, model_name=None, cors_origins=None):
    """
    Create the Flask app.

    Parameters
    ----------
    sv : an already loaded SocialVec instance. If omitted, one is created with
         ``model_name``.
    model_name : the model to load when ``sv`` is not given. Defaults to the
         ``SOCIALVEC_MODEL`` environment variable, or "2020".
    cors_origins : comma-separated allowed origins. Defaults to the
         ``SOCIALVEC_CORS_ORIGINS`` environment variable, or "*".
    """
    if sv is None:
        from socialvec.socialvec import SocialVec
        sv = SocialVec(model_name=model_name or os.environ.get("SOCIALVEC_MODEL", DEFAULT_MODEL))

    if cors_origins is None:
        cors_origins = os.environ.get("SOCIALVEC_CORS_ORIGINS", DEFAULT_CORS_ORIGINS)

    app = Flask(__name__)
    CORS(app, origins=_cors_origins(cors_origins))
    app.extensions[_EXTENSION_KEY] = {
        "sv": sv,
        # Loaded here, before any request is served, so requests never race to initialize it.
        "classifier_error": _init_classifier(sv),
    }
    app.register_blueprint(bp)
    return app


def main():
    logging.basicConfig(level=logging.INFO)
    host = os.environ.get("SOCIALVEC_HOST", DEFAULT_HOST)
    port = int(os.environ.get("PORT", DEFAULT_PORT))
    app = create_app()
    app.run(host=host, port=port)


if __name__ == '__main__':
    main()
