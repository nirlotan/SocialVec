from flask import Flask, request, jsonify
from socialvec.socialvec import SocialVec
import numpy as np

app = Flask(__name__)
sv = SocialVec()

@app.route('/validate_userid', methods=['GET'])
def validate_userid():
    userid = request.args.get('userid')
    try:
        result = sv.validate_userid(userid)
        return jsonify({'userid': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/validate_username', methods=['GET'])
def validate_username():
    username = request.args.get('username')
    try:
        result = sv.validate_username(username)
        return jsonify({'userid': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/get_screen_name', methods=['GET'])
def get_screen_name():
    userid = request.args.get('userid')
    try:
        result = sv.get_screen_name(userid)
        return jsonify({'screen_name': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/get_userid', methods=['GET'])
def get_userid():
    username = request.args.get('username')
    try:
        result = sv.get_userid(username)
        return jsonify({'userid': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/get_similar', methods=['GET'])
def get_similar():
    input_val = request.args.get('input')
    topn = int(request.args.get('topn', 10))
    try:
        df = sv.get_similar(input_val, topn)
        return df.to_json(orient='records')
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/get_embeddings', methods=['GET'])
def get_embeddings():
    entity = request.args.get('entity')
    try:
        emb = sv.get_embeddings(entity)
        return jsonify({'embeddings': emb.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/get_average_embeddings', methods=['POST'])
def get_average_embeddings():
    data = request.get_json()
    entity_list = data.get('entity_list', [])
    try:
        avg_emb, count = sv.get_average_embeddings(entity_list)
        return jsonify({'average_embeddings': avg_emb.tolist(), 'count': count})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/get_similarity', methods=['GET'])
def get_similarity():
    entity1 = request.args.get('entity1')
    entity2 = request.args.get('entity2')
    try:
        sim = sv.get_similarity(entity1, entity2)
        return jsonify({'similarity': float(sim)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    classifier_name = data.get('classifier_name')
    entity = data.get('entity')
    try:
        if not hasattr(sv, 'classifier'):
            sv.init_classifier()
        emb = sv.get_embeddings(entity)
        affiliation, pred_proba = sv.classifier.predict(classifier_name, np.array(emb))
        return jsonify({'affiliation': affiliation, 'confidence': float(pred_proba)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict_proba', methods=['POST'])
def predict_proba():
    data = request.get_json()
    classifier_name = data.get('classifier_name')
    entity = data.get('entity')
    try:
        if not hasattr(sv, 'classifier'):
            sv.init_classifier()
        emb = sv.get_embeddings(entity)
        proba = sv.classifier.predict_proba(classifier_name, np.array(emb))
        return jsonify({'probability': float(proba)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

"""
import os
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)
"""