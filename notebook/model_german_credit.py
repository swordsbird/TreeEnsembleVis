from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

best_prec = 0
depth = 9
random_state = 132
n_estimators = 150
data_table = pd.read_csv('data/german.csv')

X = data_table.drop('Creditability', axis=1).values
y = data_table['Creditability'].values
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=random_state)

sm = SMOTE(random_state=random_state)
X_train, y_train = sm.fit_resample(X_train, y_train)
r_clf = RandomForestClassifier(
    max_depth=depth, random_state=random_state, n_estimators=n_estimators)

r_clf.fit(X_train, y_train)
y_pred = r_clf.predict(X_test)
features = data_table.drop('Creditability', axis=1).columns.to_list()

print('Accuracy Score is', accuracy_score(y_test, y_pred))
print('Precision is', precision_score(y_test, y_pred))
print('Recall is', recall_score(y_test, y_pred))
print('F1-Score is', f1_score(y_test, y_pred))


class PathExtractor():
    def __init__(self, forest, X, y, X_train, y_train, features):
        self.X, self.y = X, y   # original training data

        self.features = features
        self.n_features = len(self.features)
        self.categories = np.unique(y).tolist()
        self.n_categories = len(self.categories)
        self.feature_range = [np.min(X, axis=0), np.max(X, axis=0)+1e-9]
        self.category_total = [np.sum(self.y == i)
                               for i in range(self.n_categories)]
        self.forest = forest
        self.X = X
        self.y = y
        self.X_train = X_train
        self.y_train = y_train
        self.n_examples = len(self.y)
        self.n_examples2 = len(self.y_train)
        self.n_estimators = forest.n_estimators

    def get_paths(self, min_impurity_decrease=0.0):
        self.paths = [[] for i in range(self.n_estimators)]
        self.min_impurity_decrease = min_impurity_decrease
        for i in range(self.n_estimators):
            self.dfs(i, 0, {}, np.ones(self.n_examples),
                     np.ones(self.n_examples2))
        return self.paths

    def dfs(self, i, u, feature_range, vec_examples, vec_examples2):
        tr = self.forest.estimators_[i].tree_

        def impurity_decrease(tr, u):
            N_t = tr.n_node_samples[u]
            I_t = tr.impurity[u]
            N = tr.n_node_samples[0]
            Lc = tr.children_left[u]
            Rc = tr.children_right[u]
            N_l = tr.n_node_samples[Lc]
            I_l = tr.impurity[Lc]
            N_r = tr.n_node_samples[Rc]
            I_r = tr.impurity[Rc]
            return N_t/N*(I_t-N_r/N_t*I_r-N_l/N_t*I_l)

        def cpy(m):
            return {key: m[key].copy() for key in m}

        if tr.children_left[u] < 0 or tr.children_right[u] < 0 or impurity_decrease(tr, u) < self.min_impurity_decrease:
            distribution = [np.dot(vec_examples, self.y == cid)
                            for cid in range(self.n_categories)]
            distribution2 = [np.dot(vec_examples2, self.y_train == cid)
                             for cid in range(self.n_categories)]
            output = np.argmax(distribution2)
            self.paths[i].append({
                "name": 'r%d-%d' % (len(self.paths[i]), i),
                "tree_index": i,
                "rule_index": len(self.paths[i]),
                "range": {str(key): feature_range[key].copy() for key in feature_range},
                "distribution": distribution,
                "coverage": sum(distribution),
                "sample": vec_examples,
                "output": str(output)
            })
        else:
            feature = tr.feature[u]
            threshold = tr.threshold[u]

            _feature_range = cpy(feature_range)
            if not feature in feature_range:
                _feature_range[feature] = [self.feature_range[0]
                                           [feature], self.feature_range[1][feature]+1e-9]
            _feature_range[feature][1] = min(
                _feature_range[feature][1], threshold)

            _vec_examples = vec_examples*(self.X[:, feature] <= threshold)
            _vec_examples2 = vec_examples2 * \
                (self.X_train[:, feature] <= threshold)

            self.dfs(
                i, tr.children_left[u], _feature_range, _vec_examples, _vec_examples2)

            _feature_range = cpy(feature_range)
            if not feature in feature_range:
                _feature_range[feature] = [self.feature_range[0]
                                           [feature], self.feature_range[1][feature]]
            _feature_range[feature][0] = threshold

            _vec_examples = vec_examples*(self.X[:, feature] > threshold)
            _vec_examples2 = vec_examples2 * \
                (self.X_train[:, feature] > threshold)
            self.dfs(
                i, tr.children_right[u], _feature_range, _vec_examples, _vec_examples2)

    # given X as input, find the range of fid-th feature to keep the prediction unchanged
    def getRange(self, X, fid):
        step = (self.feature_range[1][fid]-self.feature_range[0][fid])*0.005
        L, R = X[fid], X[fid]
        Xi = X.copy()
        ei = np.array([1 if i == fid else 0 for i in range(self.n_features)])
        result0 = self.predict([X])[0]
        result1 = result0

        while(result1 == result0 and L > self.feature_range[0][fid]):
            Xi = Xi-step*ei
            result1 = self.predict([Xi])[0]
            L -= step
        L = max(L, self.feature_range[0][fid])
        LC = result1

        Xi = X.copy()
        while(result1 == result0 and R < self.feature_range[1][fid]):
            Xi = Xi+step*ei
            result1 = self.predict([Xi])[0]
            R += step
        R = min(R, self.feature_range[1][fid])
        RC = result1
        return {
            "L": L,
            "LC": LC,  # the prediction when X[fid]=L-eps
            "R": R,
            "RC": RC,  # the prediction when X[fid]=R+eps
        }

rf = PathExtractor(r_clf, X, y, X_train, y_train, features)
from LP_rules_extractor import Extractor
ex = Extractor(rf, X, y)
ex.extract

paths = rf.get_paths()

all_paths = []
for t in paths:
    all_paths = all_paths + t

from annoy import AnnoyIndex
t = AnnoyIndex(len(all_paths[0]['sample']), 'euclidean')
for i in range(len(all_paths)):
    t.add_item(i, all_paths[i]['sample'])
t.build(10)
