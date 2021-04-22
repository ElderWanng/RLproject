import numpy as np

A = np.array([[ 1.    ,  0.01  ,  0.    ,  0.    ,  0.    ,  0.    ],
        [ 0.    ,  1.    ,  0.    ,  0.    , -0.0981,  0.    ],
        [ 0.    ,  0.    ,  1.    ,  0.01  ,  0.    ,  0.    ],
        [ 0.    ,  0.    ,  0.    ,  1.    ,  0.    ,  0.    ],
        [ 0.    ,  0.    ,  0.    ,  0.    ,  1.    ,  0.01  ],
        [ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  1.    ]])
B =  np.array([[ 0.   ,  0.   ],
        [-0.   , -0.   ],
        [ 0.   ,  0.   ],
        [ 0.02 ,  0.02 ],
        [ 0.   ,  0.   ],
        [ 0.015, -0.015]])
err = 1000
Q = np.diag([1, 1, 1, 1, 1, 1])
R = np.diag([0.1, 0.1])
P_last = Q


while True:
    Pnew = Q + A.transpose().dot(P_last).dot(A) - A.transpose().dot(P_last).dot(B).dot(np.linalg.inv(B.transpose().dot(P_last).dot(B)+R)).dot(B.transpose().dot(P_last).dot(A))
    err = Pnew.reshape(-1) - P_last.reshape(-1)
    err = np.sum(err @ err.T)
    if err < 0.00011:
        break
    else:
        print(err)
        P_last = Pnew

K = -np.identity(2).dot(np.linalg.inv(B.transpose().dot(P_last).dot(B)+R)).dot(B.transpose().dot(P_last).dot(A))
print(K)
