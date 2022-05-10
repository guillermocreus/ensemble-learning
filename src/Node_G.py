
class Node_G:
    def __init__(self, feature=None, values_left=None, values_right=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.values_left = values_left
        self.values_right = values_right
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None
