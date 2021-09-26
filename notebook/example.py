from os import path
from copy import deepcopy

def visit_boosting_tree(tree, path = {}):
    if 'decision_type' not in tree:
        path['value'] = tree['leaf_value']
        path['weight'] = tree['leaf_weight']
        return [{
            'range': path,
            'value': tree['leaf_value'],
            'weight': tree['leaf_weight'],
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

def assign_value_for_random_forest(paths, data):
    X, y = data
    for path in paths:
        ans = 2 * y - 1
        m = path['range']
        for key in m:
            ans = ans * (X[:, int(key)] >= m[key][0]) * (X[:, int(key)] < m[key][1])
        pos = (ans == 1).sum()
        neg = (ans == -1).sum()
        path['value'] = 1 if pos > neg else -1

def path_extractor(model, model_type, data = None):
    if model_type == 'random forest' :
        ret = []
        for estimator in model.estimators_:
            ret += path_extractor(estimator, 'decision tree')
        assign_value_for_random_forest(ret, data)
        return ret
    elif model_type == 'lightgbm':
        ret = []
        info = model._Booster.dump_model()
        for tree in info['tree_info']:
            ret += visit_boosting_tree(tree['tree_structure'])
        return ret
    elif model_type == 'decision tree':
        return visit_decision_tree(model.tree_)
    return []
    
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
#X = StandardScaler().fit_transform(X)
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
paths = path_extractor(r_clf, 'random forest', (X_train, y_train))

print(paths[0])


# 直接将paths导入Extractor
# 没有完成修改
#

import pulp
import numpy as np
from sklearn.neighbors import LocalOutlierFactor

class Extractor:
    # 可以调用的接口：compute_accuracy和extract
    def __init__(self, rf, X_train, y_train, X_test, y_test):
        # rf：随机森林模型
        # X_raw、y_raw：训练数据集
        self.rf = rf
        self.X_raw = X_train
        self.y_raw = y_train
        self.X_test = X_test
        self.y_test = y_test
        _paths = rf.get_paths()
        self.paths = [p for r in _paths for p in r]
        self.paths.sort(key=lambda x: -x['coverage'])

    def compute_accuracy_on_train(self, paths):
        # 计算数据集在给定规则集下的accuracy
        # paths：规则集，为list
        Mat = self.getMat(self.X_raw, self.y_raw, paths)
        idx = np.argwhere(np.all(Mat[..., :] == 0, axis=0))
        Mat = np.delete(Mat, idx, axis=1)
        right = np.sum(Mat, axis=0)
        return np.sum(np.where(right >= 0, 1, 0)) / len(self.X_raw)

    def compute_accuracy_on_test(self, paths):
        # 计算数据集在给定规则集下的accuracy
        # paths：规则集，为list
        Mat = self.getMat(self.X_test, self.y_test, paths)
        idx = np.argwhere(np.all(Mat[..., :] == 0, axis=0))
        Mat = np.delete(Mat, idx, axis=1)
        right = np.sum(Mat, axis=0)
        return np.sum(np.where(right >= 0, 1, 0)) / len(self.X_test)

    def extract(self, max_num, tau):
        # 根据给定的max_num和tau，使用rf的全部规则和数据集抽取出相应的规则
        # max_num：抽取出规则的最大数量
        # tau：每个样本允许的最大惩罚
        # 返回抽取出规则的列表、数据集使用全部规则的accuracy、数据集使用抽取规则的accuracy
        Mat = self.getMat(self.X_raw, self.y_raw, self.paths)
        w = self.getWeight(Mat)
        new_paths, new_path_indexes = self.LP_extraction(w, Mat, max_num, tau)
        accuracy_origin = self.compute_accuracy_on_test(self.paths)
        accuracy_new = self.compute_accuracy_on_test(new_paths)
        return new_path_indexes, new_paths, accuracy_origin, accuracy_new

    def path_score(self, path, X, y):
        ans = 2 * (y == int(path.get('output'))) - 1
        m = path.get('range')
        for key in m:
            ans = ans * (X[:, int(key)] >= m[key][0]) * (X[:, int(key)] < m[key][1])
        return ans

    def getMat(self, X_raw, y_raw, paths):
        # 覆盖矩阵Mat
        Mat = np.array([self.path_score(p, X_raw, y_raw) for p in paths]).astype('float')
        return Mat

    def getWeight(self, Mat):
        # 权重向量w
        RXMat = np.abs(Mat)
        XRMat = RXMat.transpose()
        XXAnd = np.dot(XRMat, RXMat)
        XROne = np.ones(XRMat.shape)
        XXOr = 2 * np.dot(XROne, RXMat) - XXAnd
        XXOr = (XXOr + XXOr.transpose()) / 2
        XXDis = 1 - XXAnd / XXOr
        K = int(np.ceil(np.sqrt(len(self.X_raw))))
        clf = LocalOutlierFactor(n_neighbors=K, metric="precomputed")
        clf.fit(XXDis)
        XW = -clf.negative_outlier_factor_
        MXW, mXW = np.max(XW), np.min(XW)
        XW = 1 + (3 - 1) * (XW - mXW) / (MXW - mXW)
        return XW / np.sum(XW)

    def LP_extraction(self, w, Mat, max_num, tau):
        m = pulp.LpProblem(sense=pulp.LpMinimize)
        # 创建最小化问题
        var = []
        for i in range(len(self.paths)):
            var.append(pulp.LpVariable(f'x{i}', cat=pulp.LpContinuous, lowBound=0, upBound=1))
        for i in range(len(w)):
            var.append(pulp.LpVariable(f'k{i}', cat=pulp.LpContinuous, lowBound=0))
        # 添加变量x_0至x_{M-1}, k_0至k_{N-1}

        m += pulp.lpSum([w[j] * (var[j + len(self.paths)])
                         for j in range(len(w))])
        # 添加目标函数

        m += (pulp.lpSum([var[j] for j in range(len(self.paths))]) <= max_num)
        # 筛选出不超过max_num条规则

        for j in range(len(w)):
            m += (var[j + len(self.paths)] >= 1000 + tau - pulp.lpSum([var[k] * Mat[k][j] for k in range(len(self.paths))]))
            m += (var[j + len(self.paths)] >= 1000)
            # max约束

        m.solve(pulp.PULP_CBC_CMD())#solver = pulp.solver.CPLEX())#
        new_paths = [self.paths[i] for i in range(len(self.paths)) if var[i].value() > 0]
        new_path_indexes = [self.paths[i]['name'] for i in range(len(self.paths)) if var[i].value() > 0.5]
        return new_paths, new_path_indexes
