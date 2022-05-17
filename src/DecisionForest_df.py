import numpy as np
from CART_df import CART_df

from sklearn.metrics import f1_score
from sklearn.base import BaseEstimator, ClassifierMixin

class DecisionForest_df(BaseEstimator, ClassifierMixin):
    def __init__(self, NT=100, F=-1):
        self.NT = NT
        self.decision_trees = [CART_df() for _ in range(NT)]
        
        self.F = F
        self.features_for_clfs = []
        self.importance = None
        self.df_columns = None
        
    def score(self, X, y):
        y_hat = self.predict(X)
        return f1_score(y, y_hat)
    
    def fit(self, X, y, verbose=0):
        self.importance = np.zeros(X.shape[1])
        self.df_columns = X.columns
        
        for i in range(self.NT):
            np.random.seed(i)
            if self.F == -1:
                F = 1 + np.random.randint(X.shape[1])
            else:
                F = self.F
                
            chosen_features = np.random.choice(X.shape[1], F, replace=False)
            self.features_for_clfs.append(chosen_features)
            clf = self.decision_trees[i]
            
            X_red = X.iloc[:, chosen_features]
            
            clf.fit(X_red, y)
            
            self.importance[chosen_features] += clf.importance
        
        if self.df_columns is not None and verbose:
            ordered_importance = np.flip(np.argsort(self.importance))
            print(f'Feature importance: {self.df_columns[ordered_importance]}')
        
        return self
    
    def predict(self, X):
        all_preds = []
        for i in range(self.NT):
            clf = self.decision_trees[i]
            
            X_red = X.iloc[:, self.features_for_clfs[i]]
            all_preds.append(clf.predict(X_red))
        
        all_preds = np.array(all_preds)
        preds = [np.argmax(np.bincount(all_preds[:, sample])) for sample in range(all_preds.shape[1])]
        preds = np.array(preds)
        return preds
        