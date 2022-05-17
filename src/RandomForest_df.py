import numpy as np
from CART_df import CART_df
from sklearn.metrics import f1_score
from sklearn.base import BaseEstimator, ClassifierMixin

class RandomForest_df(BaseEstimator, ClassifierMixin):
    def __init__(self, NT=100, F=1):
        self.NT = NT
        
        self.F = F
        self.decision_trees = [CART_df(F=F, method='RF') for _ in range(NT)]
        self.features_for_clfs = []
        self.importance = None
        self.df_columns = None
        
    def score(self, X, y):
        y_hat = self.predict(X)
        return f1_score(y, y_hat)
    
    def fit(self, X, y, seed=2, verbose=0):
        self.importance = np.zeros(X.shape[1])
        self.df_columns = X.columns
        np.random.seed(seed)
        for i in range(self.NT):
            clf = self.decision_trees[i]
            
            bootstrap_samples = np.random.choice(X.shape[0], X.shape[0], replace=True)
            
            X_bootstrap = X.iloc[bootstrap_samples]
            X_bootstrap.reset_index(drop=True, inplace=True)
            
            y_bootstrap = y.iloc[bootstrap_samples]
            y_bootstrap.reset_index(drop=True, inplace=True)
            
            clf.fit(X_bootstrap, y_bootstrap)
            
            self.importance += clf.importance
        
        if self.df_columns is not None and verbose:
            ordered_importance = np.flip(np.argsort(self.importance))
            print(f'Feature importance: {self.df_columns[ordered_importance]}')
        return self
    
    def predict(self, X):
        all_preds = []
        for i in range(self.NT):
            clf = self.decision_trees[i]
            
            # X_red = X.iloc[:, self.features_for_clfs[i]]
            all_preds.append(clf.predict(X))
        
        all_preds = np.array(all_preds)
        preds = [np.argmax(np.bincount(all_preds[:, sample])) for sample in range(all_preds.shape[1])]
        preds = np.array(preds)
        return preds
        