import itertools
import numpy as np
from Node_G import Node_G

class CART:
    def __init__(self, max_depth=100, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.all_partitions = {}
        
        self.importance = None
        
    def _get_partitions(self, n):
        if n in self.all_partitions:
            return self.all_partitions[n]
        
        lst = set(tuple(i) for i in itertools.product([0, 1], repeat=n))
        lst_clean = set()
        for elem in lst:
            if tuple(1 - np.array(elem)) not in lst_clean:
                lst_clean.add(elem)

        try: lst_clean.remove((0,)*n)
        except: pass    

        try: lst_clean.remove((1,)*n)
        except: pass    

        self.all_partitions[n] = lst_clean
        return self.all_partitions[n]

    def _is_finished(self, depth):
        if (depth >= self.max_depth
            or self.n_class_labels == 1
            or self.n_samples < self.min_samples_split):
            return True
        return False
    
    def _create_split_G(self, X, values_left, values_right):
        left_idx = np.argwhere(np.isin(X, values_left)).flatten()
        right_idx = np.argwhere(np.isin(X, values_right)).flatten()
        return left_idx, right_idx
    
    def Gini_X(self, X, y):
        G = 1
        for cls in np.unique(y):
            G -= ((y == cls).sum() / len(X))**2
        return G

    def Gini_X_A(self, X, y, current_split):
        values1, values2 = current_split
        
        ind1 = np.argwhere(np.isin(X, values1)).flatten()
        X1 = X[ind1]
        y1 = y[ind1]
        
        ind2 = np.argwhere(np.isin(X, values2)).flatten()
        X2 = X[ind2]
        y2 = y[ind2]
        
        return len(X1) / len(X) * self.Gini_X(X1, y1) + len(X2) / len(X) * self.Gini_X(X2, y2)
    
    def _best_split_G(self, X, y, features):
        best_split = {'score': np.inf, 
                      'feat': None, 
                      'values_left': None, 
                      'values_right': None}
        
        for feat in features:
            X_feat = X[:, feat]
            elems_col = np.unique(X_feat)
            partitions = self._get_partitions(len(elems_col))
            
            for partition in partitions:
                ind1 = np.array(partition, dtype=bool)
                values_left, values_right = elems_col[ind1], elems_col[~ind1]
                current_split = (values_left, values_right)
                score = self.Gini_X_A(X_feat, y, current_split)
                # print(current_split, score)
                
                if score < best_split['score']:
                    best_split['score'] = score
                    best_split['feat'] = feat
                    best_split['values_left'] = values_left
                    best_split['values_right'] = values_right

        return best_split['feat'], best_split['values_left'], best_split['values_right']

    def _build_tree_G(self, X, y, depth=0):
        self.n_samples, self.n_features = X.shape
        self.n_class_labels = len(np.unique(y))

        # stopping criteria
        # Note: condition (X == X[0]).all() is used to remove amibguity
        if self._is_finished(depth) or (X == X[0]).all():
            most_common_Label = np.argmax(np.bincount(y))
            return Node_G(value=most_common_Label)

        # get best split
        rnd_feats = np.random.choice(self.n_features, self.n_features, replace=False)
        best_feat, best_values_left, best_values_right = self._best_split_G(X, y, rnd_feats)
        # print(f'Best split {best_feat, best_values_left, best_values_right}')
        
        # grow children recursively
        self.importance[best_feat] += 1
        left_idx, right_idx = self._create_split_G(X[:, best_feat], best_values_left, best_values_right)
        left_child = self._build_tree_G(X[left_idx, :], y[left_idx], depth + 1)
        right_child = self._build_tree_G(X[right_idx, :], y[right_idx], depth + 1)
        return Node_G(best_feat, best_values_left, best_values_right, left_child, right_child)
    
    def _traverse_tree_G(self, x, node):
        if node.is_leaf():
            return node.value
        
        if x[node.feature] in node.values_left:
            return self._traverse_tree_G(x, node.left)
        return self._traverse_tree_G(x, node.right)

    def fit(self, X, y):
        self.importance = np.zeros(X.shape[1])
        self.root = self._build_tree_G(X, y)

    def predict(self, X):
        predictions = [self._traverse_tree_G(x, self.root) for x in X]
        return np.array(predictions)
    