import os
import utils
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

class RandomForestDemo(RandomForestClassifier):
    # features: (list) name of features
    # n_features: (int) number of features
    # feature_range[0:1][fid]: range of the fid-th feature
    # X,y: input data
    # categories: (list) name of categories
    # n_categories: (int) number of categories
    # node2path[tid][nid]=pid: converts the node_id of the tid-th tree to path_id 
    # distribution[tid][pid][cid]: number of examples in the tid-th tree, pid-th path, in cid-th category
    # paths[tid][pid][0:1]: the range of pid-th path in tid-th tree
    # rule_example_table[tid][pid][xid]: True if the pid-th path of tid-th tree covers xid-th example(X[xid])
    # other attributes that inherits sklearn.ensemble.RandomForestClassifier
    def __init__(self,reader=utils.reader,sampling_method="SMOTE",**kwargs):
        RandomForestClassifier.__init__(self,**kwargs)
        
        X,y = reader.getData()
        self.X,self.y=X,y   # original training data

        self.reader=reader
        self.features=reader.features
        self.n_features=len(self.features)
        self.categories=reader.categories
        self.n_categories=len(self.categories)
        
        self.n_examples=len(self.y)
        self.feature_range=[np.min(X,axis=0),np.max(X,axis=0)+1e-9]

        self.sampling_method=sampling_method
        if(sampling_method=="SMOTE"):
            sm=SMOTE(random_state=42)
            self.SX,self.Sy=sm.fit_resample(self.X,self.y)
        else:
            self.SX,self.Sy=self.X,self.y
        self.n_examples2=len(self.Sy)

        self.fit(self.SX, self.Sy)

        self.category_total=[np.sum(self.y==i) for i in range(self.n_categories)]

    def get_paths(self,min_impurity_decrease=0.0):
        self.paths=[[] for i in range(self.n_estimators)]
        self.min_impurity_decrease=min_impurity_decrease
        for i in range(self.n_estimators):
            self.dfs(i,0,{},np.ones(self.n_examples),np.ones(self.n_examples2))
        return self.paths

    def dfs(self,i,u,feature_range,vec_examples,vec_examples2):
        tr=self.estimators_[i].tree_
        def impurity_decrease(tr,u):
            N_t=tr.n_node_samples[u]
            I_t=tr.impurity[u]
            N=tr.n_node_samples[0]
            Lc=tr.children_left[u]
            Rc=tr.children_right[u]
            N_l=tr.n_node_samples[Lc]
            I_l=tr.impurity[Lc]
            N_r=tr.n_node_samples[Rc]
            I_r=tr.impurity[Rc]
            return N_t/N*(I_t-N_r/N_t*I_r-N_l/N_t*I_l)
        def cpy(m):
            return {key:m[key].copy() for key in m}
            
        if tr.children_left[u]<0 or tr.children_right[u]<0 or impurity_decrease(tr,u)<self.min_impurity_decrease:
            distribution=[np.dot(vec_examples,self.y==cid) for cid in range(self.n_categories)]
            distribution2=[np.dot(vec_examples2,self.Sy==cid) for cid in range(self.n_categories)]
            output=np.argmax(distribution2)
            self.paths[i].append({
                "name":'r%d-%d'%(len(self.paths[i]),i),
                "tree_index":i,
                "rule_index":len(self.paths[i]),
                "range":{str(key):feature_range[key].copy() for key in feature_range},
                "distribution":distribution,
                "coverage":sum(distribution),
                "output":str(output)
            })
        else:
            feature=tr.feature[u]
            threshold=tr.threshold[u]

            _feature_range=cpy(feature_range)
            if not feature in feature_range:
                _feature_range[feature]=[self.feature_range[0][feature],self.feature_range[1][feature]+1e-9]
            _feature_range[feature][1]=min(_feature_range[feature][1],threshold)

            _vec_examples = vec_examples*(self.X[:,feature]<=threshold)
            _vec_examples2 = vec_examples2*(self.SX[:,feature]<=threshold)
            
            self.dfs(i,tr.children_left[u],_feature_range,_vec_examples,_vec_examples2)
            
            _feature_range=cpy(feature_range)
            if not feature in feature_range:
                _feature_range[feature]=[self.feature_range[0][feature],self.feature_range[1][feature]]
            _feature_range[feature][0]=threshold
            
            _vec_examples = vec_examples*(self.X[:,feature]>threshold)
            _vec_examples2 = vec_examples2*(self.SX[:,feature]>threshold)
            self.dfs(i,tr.children_right[u],_feature_range,_vec_examples,_vec_examples2)
    
    # given X as input, find the range of fid-th feature to keep the prediction unchanged
    def getRange(self,X,fid):
        step=(self.feature_range[1][fid]-self.feature_range[0][fid])*0.005
        L,R=X[fid],X[fid]
        Xi=X.copy()
        ei=np.array([1 if i==fid else 0 for i in range(self.n_features)])
        result0=self.predict([X])[0]
        result1=result0

        while(result1==result0 and L>self.feature_range[0][fid]):
            Xi=Xi-step*ei
            result1=self.predict([Xi])[0]
            L-=step
        L=max(L,self.feature_range[0][fid])
        LC=result1

        Xi=X.copy()
        while(result1==result0 and R<self.feature_range[1][fid]):
            Xi=Xi+step*ei
            result1=self.predict([Xi])[0]
            R+=step
        R=min(R,self.feature_range[1][fid])
        RC=result1
        return {
            "L":L,
            "LC":LC,  # the prediction when X[fid]=L-eps
            "R":R,
            "RC":RC,  # the prediction when X[fid]=R+eps
        }

if __name__ == "__main__":
    demo=RandomForestDemo(reader=utils.reader)
