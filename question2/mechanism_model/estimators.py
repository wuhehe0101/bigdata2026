import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin

class MonotoneLogistic(ClassifierMixin, BaseEstimator):
 """Weighted ridge logistic regression with nonnegative standardized slopes."""
 def __init__(self, C=.1): self.C=C
 def fit(self,X,y,sample_weight=None):
  X=np.asarray(X,dtype=float);y=np.asarray(y);w=np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight)
  def fun(v):
   z=v[0]+X@v[1:];loss=np.sum(w*(np.logaddexp(0,z)-y*z))+.5*np.dot(v[1:],v[1:])/self.C
   err=w*(expit(z)-y);grad=np.r_[err.sum(),X.T@err+v[1:]/self.C];return loss,grad
  result=minimize(fun,np.zeros(X.shape[1]+1),jac=True,method='L-BFGS-B',bounds=[(None,None)]+[(0,None)]*X.shape[1],options={'maxiter':3000,'ftol':1e-12,'gtol':1e-7})
  assert result.success,result.message
  self.intercept_=result.x[:1];self.coef_=result.x[1:][None,:];self.classes_=np.array([0,1]);self.n_features_in_=X.shape[1];return self
 def predict_proba(self,X):
  p=expit(np.asarray(X)@self.coef_[0]+self.intercept_[0]);return np.c_[1-p,p]

