import numpy as np
from CART import CART
from sklearn.metrics import f1_score
from sklearn.base import BaseEstimator, ClassifierMixin

class RandomForest(BaseEstimator, ClassifierMixin):
    def __init__(self, NT=100, F=1, df_columns=None):
        self.NT = NT
        self.decision_trees = [CART() for _ in range(NT)]
        
        self.F = F
        self.features_for_clfs = []
        self.importance = None
        self.df_columns = df_columns
        
    def score(self, X, y):
        y_hat = self.predict(X)
        return f1_score(y, y_hat)
    
    def fit(self, X, y, seed=2, verbose=0):
        self.importance = np.zeros(X.shape[1])
        
        np.random.seed(seed)
        for i in range(self.NT):
            chosen_features = np.random.choice(X.shape[1], self.F, replace=False)
            self.features_for_clfs.append(chosen_features)
            clf = self.decision_trees[i]
            
            bootstrap_samples = np.random.choice(X.shape[0], X.shape[0])
            X_bootstrap = X[bootstrap_samples, :][:, chosen_features]
            y_bootstrap = y[bootstrap_samples]
            
            clf.fit(X_bootstrap, y_bootstrap)
            
            self.importance[chosen_features] += clf.importance
        
        if self.df_columns is not None and verbose:
            ordered_importance = np.flip(np.argsort(self.importance))
            print(f'Feature importance: {self.df_columns[ordered_importance]}')
        return self
    
    def predict(self, X):
        all_preds = []
        for i in range(self.NT):
            clf = self.decision_trees[i]
            
            X_red = X[:, self.features_for_clfs[i]]
            all_preds.append(clf.predict(X_red))
        
        all_preds = np.array(all_preds)
        preds = [np.argmax(np.bincount(all_preds[:, sample])) for sample in range(all_preds.shape[1])]
        preds = np.array(preds)
        return preds
        