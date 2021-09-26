from os import path
from copy import deepcopy
import numpy as np

def visit_boosting_tree(tree, path = {}):
    if 'decision_type' not in tree:
        return [{
            'range': path,
            'value': tree['leaf_value'],
            'weight': tree['leaf_weight'],
            'confidence': 1,
        }]
    
    key = tree['split_feature']
    thres = tree['threshold']
    ret = []
    leftpath = deepcopy(path)
    if key in leftpath:
        r = leftpath[key]
        leftpath[key] = [r[0], min(r[1], thres)]
    else:
        leftpath[key] = [-1e9, thres]
    ret += visit_boosting_tree(tree['left_child'], leftpath)

    rightpath = deepcopy(path)
    if key in rightpath:
        r = rightpath[key]
        rightpath[key] = [max(r[0], thres), r[1]]
    else:
        rightpath[key] = [thres, 1e9]
    ret += visit_boosting_tree(tree['right_child'], rightpath)

    return ret

def visit_decision_tree(tree, index = 0, path = {}):
    if tree.children_left[index] == -1 and tree.children_right[index] == -1:
        return [{
            'range': path,
            'value': 0,
            'weight': 1,
        }]
    key = tree.feature[index]
    thres = tree.threshold[index]
    ret = []
    leftpath = deepcopy(path)
    if key in leftpath:
        r = leftpath[key]
        leftpath[key] = [r[0], min(r[1], thres)]
    else:
        leftpath[key] = [-1e9, thres]
    ret += visit_decision_tree(tree, tree.children_left[index], leftpath)
    
    rightpath = deepcopy(path)
    if key in rightpath:
        r = rightpath[key]
        rightpath[key] = [max(r[0], thres), r[1]]
    else:
        rightpath[key] = [thres, 1e9]
    ret += visit_decision_tree(tree, tree.children_right[index], rightpath)

    return ret

def assign_value_for_random_forest(paths, model, data):
    X, y = data
    for path in paths:
        ans = y * 2 - 1
        m = path['range']
        for key in m:
            ans = ans * (X[:, int(key)] >= m[key][0]) * (X[:, int(key)] < m[key][1])
        index = np.nonzero(ans)[0]
        if len(index) == 0:
            path['value'] = None
            continue
        index = index[0]
        value = model.estimators_[path['tree']].predict(X[index: index + 1])
        value = value[0]
        pos = np.sum(ans > 0)
        neg = np.sum(ans < 0)
        path['value'] = 1 if value > 0 else -1
        path['confidence'] = max(pos / (pos + neg), neg / (pos + neg))
        if pos == neg:
            path['value'] = 0

def path_extractor(model, model_type, data = None):
    if model_type == 'random forest' :
        ret = []
        for index, estimator in enumerate(model.estimators_):
            paths = path_extractor(estimator, 'decision tree')
            for path in paths:
                path['tree'] = index
            ret += paths
        assign_value_for_random_forest(ret, model, data)
        ret = [x for x in ret if x['value'] != None]
        return ret
    elif model_type == 'lightgbm':
        ret = []
        info = model._Booster.dump_model()
        for index, tree in enumerate(info['tree_info']):
            paths = visit_boosting_tree(tree['tree_structure'])
            for path in paths:
                path['tree'] = index
            ret += paths
        return ret
    elif model_type == 'decision tree':
        return visit_decision_tree(model.tree_)
    return []
    
def path_score(path, X):
    ans = int(path.get('value'))
    m = path.get('range')
    for key in m:
        ans = ans * (X[:, int(key)] >= m[key][0]) * (X[:, int(key)] < m[key][1])
    return ans

def predict(paths, X):
    Mat = np.array([path_score(p, X) for p in paths]).astype('float')
    right = np.sum(Mat, axis=0)
    return np.where(right >= 0, 1, 0)

def predict_original(paths, X):
    Mat = np.array([path_score(p, X) for p in paths]).astype('float')
    right = np.sum(Mat, axis=0)
    return right

def compute_accuracy(paths, X, ground_truth):
    Mat = np.array([path_score(p, X) for p in paths]).astype('float')
    #idx = np.argwhere(np.all(Mat[..., :] == 0, axis=0))
    #Mat = np.delete(Mat, idx, axis=1)
    right = np.sum(Mat, axis=0)
    return np.sum(np.where(right >= 0, 1, 0) == ground_truth) / len(X)