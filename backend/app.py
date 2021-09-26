from flask import Flask, render_template, jsonify, request
from random import *
import random
from flask_cors import CORS
import json
import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import pairwise_distances
from annoy import AnnoyIndex

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

app = Flask(__name__,
            static_folder = "./dist/static",
            template_folder = "./dist")
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

class DataLoader():
    def __init__(self, data, model, target):
        self.data_table = data
        self.model = model
        self.paths = self.model['paths']
        self.shap_values = self.model['shap_values']
        self.path_index = {}
        for index, path in enumerate(self.paths):
            self.path_index[path['name']] = index
        self.selected_indexes = self.model['selected'][:75]
        self.features = self.model['features']
        self.X = self.data_table.drop(target, axis=1).values
        self.y = self.data_table[target].values

        path_mat = np.array([path['sample'] for path in self.paths])
        tree = AnnoyIndex(len(path_mat[0]), 'euclidean')
        for i in range(len(path_mat)):
            tree.add_item(i, path_mat[i])
        tree.build(10)
        self.tree = tree

        path_dist = pairwise_distances(X = path_mat)
        K = int(np.ceil(np.sqrt(len(self.X))))
        clf = LocalOutlierFactor(n_neighbors=K, metric="precomputed")
        clf.fit(path_dist)
        path_lof = -clf.negative_outlier_factor_

        for i in range(len(self.paths)):
            self.paths[i]['lof'] = float(path_lof[i])
            self.paths[i]['represent'] = False
        for i in self.selected_indexes:
            path = self.paths[self.path_index[i]]
            path['represent'] = True

data = pd.read_csv('../model/data/german.csv')
model = pickle.load(open('../model/output/german.pkl', 'rb'))
loader = DataLoader(data, model, 'Creditability')

@app.route('/api/data_table', methods=["POST"])
def get_data():
    data2 = pd.read_csv('../model/data/german1.csv')
    response = [[feature, data2[feature].values] for feature in data2.columns]
    return json.dumps(response, cls=NpEncoder)

@app.route('/api/samples', methods=["POST"])
def get_samples():
    data = json.loads(request.get_data(as_text=True))
    ids = data['ids']
    response = []
    for i in ids:
        response.append({
            'x': loader.X[i].tolist(),
            'y': str(loader.y[i]),
        })
    return jsonify(response)

@app.route('/api/features')
def get_features():
    return json.dumps(loader.features, cls=NpEncoder)

@app.route('/api/explore_rules', methods=["POST"])
def get_explore_rules():
    data = json.loads(request.get_data(as_text=True))
    idxs = data['idxs']
    N = data['N']
    K = int(N / len(idxs)) + 3
    response = []
    nns = []
    for name in idxs:
        j = loader.path_index[name]
        neighbors = loader.tree.get_nns_by_item(j, K)
        neighbors = [i for i in neighbors if not loader.paths[i]['represent']]
        nns += [j] + neighbors
    nns_set = set()
    for i in nns:
        if i in nns_set:
            continue
        nns_set.add(i)
        path = loader.paths[i]
        response.append({
            'name': path['name'],
            'tree_index': path['tree_index'],
            'rule_index': path['rule_index'],
            'represent': path['represent'],
            'range': path['range'],
            'LOF': path['lof'],
            'distribution': path['distribution'],
            'coverage': path['coverage'] / len(loader.X),
            'output': path['output'],
            'samples': np.flatnonzero(path['sample']).tolist(),
        })
    response = response[:N]
    return json.dumps(response, cls=NpEncoder)

@app.route('/api/rule_samples', methods=["POST"])
def get_relevant_samples():
    data = json.loads(request.get_data(as_text=True))
    names = data['names']
    N = data['N']
    vec = np.zeros(loader.X.shape[0])
    for name in names:
        vec += loader.paths[loader.path_index[name]]['sample']
    ids = np.flatnonzero(vec).tolist()
    ids = random.sample(ids, N)
    response = []
    for i in ids:
        response.append({
            'id': i,
            'x': loader.X[i].tolist(),
            'y': str(loader.y[i]),
            'shap_values': loader.shap_values[i].values,
        })
    return json.dumps(response, cls=NpEncoder)

@app.route('/api/selected_rules')
def get_selected_rules():
    response = []
    for i in loader.selected_indexes:
        path = loader.paths[loader.path_index[i]]
        response.append({
            'name': path['name'],
            'tree_index': path['tree_index'],
            'rule_index': path['rule_index'],
            'represent': path['represent'],
            'range': path['range'],
            'LOF': path['lof'],
            'distribution': path['distribution'],
            'coverage': path['coverage'] / len(loader.X),
            'output': path['output'],
            'samples': np.flatnonzero(path['sample']).tolist(),
        })
    return json.dumps(response, cls=NpEncoder)

'''
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    if app.debug:
        return requests.get('http://localhost:8080/{}'.format(path)).text
    return render_template("index.html")
'''