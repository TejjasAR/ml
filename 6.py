import numpy as np
import matplotlib.pyplot as plt

X=np.arange(3,33).reshape(-1,1)

y=np.array([1,2,1,2,1,3,4,5,4,5,
            6,5,6,7,8,9,10,11,11,12,
            11,11,10,12,11,11,10,9,8,8])

tau=0.5

Xt=np.arange(0,40,0.1).reshape(-1,1)

pred=[]

for x in Xt:
    w=np.exp(-(X-x)**2/(2*tau**2)).ravel()
    W=np.diag(w)

    theta=np.linalg.pinv(X.T@W@X)@X.T@W@y
    pred.append(x@theta)

plt.scatter(X,y,color='orange')
plt.plot(Xt,pred,'black')
plt.show()