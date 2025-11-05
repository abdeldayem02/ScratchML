import numpy
import cvxopt


class SupportVectorMachine(object):

    def __init__(self, C=1.0, kernel='linear', gamma='scale', degree=3):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.alpha = None
        self.w = None
        self.b = None
        self.support_vectors = None
          
    def _kernel(self):
        
        
        pass

    def _compute_margin(self):
        
        pass

    def _decision_function(self):
        
        pass

    def fit(self, X, y):
        
        pass

    def predict(self, X):
        
        pass

    
