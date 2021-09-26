import pulp
import numpy as np
from sklearn.neighbors import LocalOutlierFactor


class Extractor:
    # 可以调用的接口：compute_accuracy和extract
    def __init__(self, paths, X_raw, y_raw):
        # paths：规则集
        # X_raw、y_raw：训练数据集
        self.X_raw = X_raw
        self.y_raw = y_raw
        self.paths = paths
        self.paths.sort(key=lambda x: -x['coverage'])

    def compute_accuracy(self, paths):
        # 计算数据集在给定规则集下的accuracy
        # paths：规则集，为list
        Mat = self.getMat(self.X_raw, self.y_raw, paths)
        idx = np.argwhere(np.all(Mat[..., :] == 0, axis=0))
        Mat = np.delete(Mat, idx, axis=1)
        right = np.sum(Mat, axis=0)
        return np.sum(np.where(right >= 0, 1, 0)) / len(self.X_raw)

    def extract(self, max_num, tau):
        # 根据给定的max_num和tau，使用rf的全部规则和数据集抽取出相应的规则
        # max_num：抽取出规则的最大数量
        # tau：每个样本允许的最大惩罚
        # 返回抽取出规则的列表、数据集使用全部规则的accuracy、数据集使用抽取规则的accuracy
        Mat = self.getMat(self.X_raw, self.y_raw, self.paths)
        w = self.getWeight(Mat)
        new_paths = self.LP_extraction(w, Mat, max_num, tau)
        accuracy_origin = self.compute_accuracy(self.paths)
        accuracy_new = self.compute_accuracy(new_paths)
        return new_paths, accuracy_origin, accuracy_new

    def __path_score(self, path, X, y):
        ans = 2 * (y == int(path.get('output'))) - 1
        m = path.get('range')
        for key in m:
            ans = ans * (X[:, int(key)] >= m[key][0]) * (X[:, int(key)] < m[key][1])
        return ans

    def __getMat(self, X_raw, y_raw, paths):
        # 覆盖矩阵Mat
        Mat = np.array([self.path_score(p, X_raw.values, y_raw.values) for p in paths]).astype('float')
        return Mat

    def __getWeight(self, Mat):
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
        XW = 1 + (5 - 1) * (XW - mXW) / (MXW - mXW)
        return XW / np.sum(XW)

    def __LP_extraction(self, w, Mat, max_num, tau):
        # M rules, N samples
        # 输入
        # Mat: 规则覆盖样本的 M * N 矩阵
        # w: N 维权重向量
        # paths: 全部 M 条规则
        # max_num: 允许选出的最大规则数目
        # tau: 允许的最大惩罚
        # 输出
        # 返回筛选出的规则
        m = pulp.LpProblem(sense=pulp.LpMinimize)
        # 创建最小化问题
        var = []
        for i in range(len(self.paths)):
            var.append(pulp.LpVariable(f'x{i}', cat=pulp.LpBinary))
        for i in range(len(w)):
            var.append(pulp.LpVariable(f'k{i}', cat=pulp.LpInteger, lowBound=0))
        # 添加变量x_0至x_{M-1}, k_0至k_{N-1}

        m += pulp.lpSum([w[j] * (var[j + len(self.paths)])
                         for j in range(len(w))])
        # 添加目标函数

        m += (pulp.lpSum([var[j] for j in range(len(self.paths))]) <= max_num)
        # 筛选出不超过max_num条规则

        for j in range(len(w)):
            m += (var[j + len(self.paths)] >= tau - pulp.lpSum([var[k] * Mat[k][j] for k in range(len(self.paths))]))
            # max约束

        m.solve(pulp.PULP_CBC_CMD())
        return [self.paths[i] for i in range(len(self.paths)) if var[i].value() > 0]
