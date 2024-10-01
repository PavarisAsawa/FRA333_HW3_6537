from HW3_utils import *
import numpy as np
from FRA333_HW3_6537 import *
# import roboticstoolbox as rtb

q = [0.8,2.3,1.14]

# J_full = endEffectorJacobianHW3(q)
# J_reduce = J_full[0:3,0:3]
# det = np.linalg.det(J_reduce)
# print(det)


w = [1,2,3,4,5,6]

a = computeEffortHW3(q,w)
# print(a.shape)