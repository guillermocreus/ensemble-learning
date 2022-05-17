import itertools
import numpy as np
from Node_G_df import Node_G_df

class CART_df:
    def __init__(self, min_samples_split=2, F=None, method='DF'):
        self.min_samples_split = min_samples_split
        self.root = None
        self.all_partitions = {}
        self.method = method
        self.F = F
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

    def _is_finished(self):
        if (self.n_class_labels == 1
            or self.n_samples < self.min_samples_split):
            return True
        return False
    
    def _create_split_str(self, X, values_left, values_right):
        left_idx = X.isin(values_left)
        right_idx = X.isin(values_right)
        return left_idx, right_idx
    
    def _create_split_num(self, X, thr):
        left_idx = X <= thr
        right_idx = X > thr
        return left_idx, right_idx
    
    def Gini_X(self, X, y):
        G = 1
        for cls in np.unique(y):
            G -= ((y == cls).sum() / len(X))**2
        return G

    def Gini_X_A_str(self, X, y, current_split):
        values1, values2 = current_split
        ind1, ind2 = self._create_split_str(X, values1, values2)
        X1, y1 = X[ind1], y[ind1]    
        X2, y2 = X[ind2], y[ind2]
        
        return len(X1) / len(X) * self.Gini_X(X1, y1) + len(X2) / len(X) * self.Gini_X(X2, y2)

    def Gini_X_A_num(self, X, y, thr):
        ind1, ind2 = self._create_split_num(X, thr)
        X1, y1 = X[ind1], y[ind1]    
        X2, y2 = X[ind2], y[ind2]
        
        return len(X1) / len(X) * self.Gini_X(X1, y1) + len(X2) / len(X) * self.Gini_X(X2, y2) 
    
    def _best_split_G(self, X, y, features):
        best_split = {'score': np.inf, 
                      'feat': None, 
                      'values_left': None, 
                      'values_right': None, 
                      'thr': None}
        
        for feat in features:
            X_feat = X.iloc[:, feat]
            elems_col = np.unique(X_feat)
            
            if X_feat.dtype not in [int, float]:
                partitions = self._get_partitions(len(elems_col))
                
                for partition in partitions:
                    ind1 = np.array(partition, dtype=bool)
                    values_left, values_right = elems_col[ind1], elems_col[~ind1]
                    current_split = (values_left, values_right)
                    score = self.Gini_X_A_str(X_feat, y, current_split)
                    # print(current_split, score)
                    
                    if score < best_split['score']:
                        best_split['score'] = score
                        best_split['feat'] = feat
                        best_split['values_left'] = values_left
                        best_split['values_right'] = values_right
                        best_split['thr'] = None
            else:
                # Mid points of unique elements (elems_col <-- sorted array with the unique column elements)
                thresholds = 0.5 * (elems_col[:-1] + elems_col[1:])
                for thr in thresholds:
                    score = self.Gini_X_A_num(X_feat, y, thr)
                    if score < best_split['score']:
                        best_split['score'] = score
                        best_split['feat'] = feat
                        best_split['values_left'] = None
                        best_split['values_right'] = None
                        best_split['thr'] = thr

        return best_split['feat'], best_split['values_left'], best_split['values_right'], best_split['thr']

    def _build_tree_G(self, X, y):
        self.n_samples, self.n_features = X.shape
        self.n_class_labels = len(np.unique(y))

        # choose features
        if self.method == 'RF':
            rnd_feats = np.random.choice(self.n_features, self.F, replace=False)
        else:
            rnd_feats = np.random.choice(self.n_features, self.n_features, replace=False)

        # stopping criteria
        # Note: condition (X == X[0]).all() is used to remove ambiguity
        for feat in rnd_feats:
            X_aux = X.iloc[:, feat]
            if (X_aux == X_aux.iloc[0]).all().all():
                most_common_Label = np.argmax(np.bincount(y))
                return Node_G_df(value=most_common_Label)
        
        if self._is_finished():
            most_common_Label = np.argmax(np.bincount(y))
            return Node_G_df(value=most_common_Label)
            
        # get best split
        best_feat, best_values_left, best_values_right, best_thr = self._best_split_G(X, y, rnd_feats)
        # print(f'Best split {best_feat, best_values_left, best_values_right}')
        
        # grow children recursively
        self.importance[best_feat] += 1
        
        if best_thr is None:          
            left_idx, right_idx = self._create_split_str(X.iloc[:, best_feat], best_values_left, best_values_right)
        else:
            left_idx, right_idx = self._create_split_num(X.iloc[:, best_feat], best_thr)
        
        left_child = self._build_tree_G(X[left_idx], y[left_idx])
        right_child = self._build_tree_G(X[right_idx], y[right_idx])
        
        return Node_G_df(best_feat, best_values_left, best_values_right, best_thr, left_child, right_child)
    
    def _traverse_tree_G(self, x, node):
        if node.is_leaf():
            return node.value
        
        if node.thr is not None:
            if (x.iloc[:, node.feature] <= node.thr).all():
                return self._traverse_tree_G(x, node.left)
            return self._traverse_tree_G(x, node.right)
                 
        if x.iloc[:, node.feature].isin(node.values_left).all():
            return self._traverse_tree_G(x, node.left)
        return self._traverse_tree_G(x, node.right)

    def fit(self, X, y):
        self.importance = np.zeros(X.shape[1])
        self.root = self._build_tree_G(X, y)

    def predict(self, X):
        predictions = []
        for ind in range(len(X)):
            row = X.iloc[[ind]]
            predictions.append(self._traverse_tree_G(row, self.root))
        return np.array(predictions)
    