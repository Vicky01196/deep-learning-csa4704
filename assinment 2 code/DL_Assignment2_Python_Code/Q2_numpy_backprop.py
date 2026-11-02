import numpy as np
np.random.seed(42)
def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-40,40)))
def forward(X,W1,b1,W2,b2):
 z1=X@W1+b1; a1=np.maximum(z1,0); p=sigmoid(a1@W2+b2); return z1,a1,p
def loss(y,p):
 p=np.clip(p,1e-12,1-1e-12); return -np.mean(y*np.log(p)+(1-y)*np.log(1-p))
def backward(X,y,z1,a1,p,W2):
 d2=(p-y)/len(X); d1=(d2@W2.T)*(z1>0)
 return {'W1':X.T@d1,'b1':d1.sum(0),'W2':a1.T@d2,'b2':d2.sum(0)}
def numeric(X,y,P,eps):
 G={}
 for k in P:
  G[k]=np.zeros_like(P[k])
  for ix in np.ndindex(P[k].shape):
   old=P[k][ix]; P[k][ix]=old+eps; lp=loss(y,forward(X,*[P[t] for t in ('W1','b1','W2','b2')])[2])
   P[k][ix]=old-eps; lm=loss(y,forward(X,*[P[t] for t in ('W1','b1','W2','b2')])[2])
   P[k][ix]=old; G[k][ix]=(lp-lm)/(2*eps)
 return G
def relerr(a,b): return np.linalg.norm(a-b)/(np.linalg.norm(a)+np.linalg.norm(b)+1e-12)
if __name__=='__main__':
 X=np.random.randn(5,4); y=np.array([[0.],[1.],[0.],[1.],[1.]])
 P={'W1':np.random.randn(4,3)*.1,'b1':np.zeros((1,3)),
    'W2':np.random.randn(3,1)*.1,'b2':np.zeros((1,1))}
 z,a,p=forward(X,*[P[k] for k in ('W1','b1','W2','b2')]); A=backward(X,y,z,a,p,P['W2'])
 for e in [1e-5,1e-6]: print('eps',e,{k:relerr(A[k],numeric(X,y,P,e)[k]) for k in P})
 # Deliberate bug: omitted ReLU derivative
 d2=(p-y)/len(X); d1=d2@P['W2'].T
 bug={'W1':X.T@d1,'b1':d1.sum(0)}
 print('Bug relative errors:',{k:relerr(A[k],bug[k]) for k in bug})
