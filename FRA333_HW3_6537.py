# file สำหรับเขียนคำตอบ
# ในกรณีที่มีการสร้าง function อื่น ๆ ให้ระบุว่า input-output คืออะไรด้วย
from HW3_utils import FKHW3
import numpy as np
# '''
# ชื่อ_รหัส(ธนวัฒน์_6461)
# 1.
# 2.
# 3.
# '''
#=============================================<คำตอบข้อ 1>======================================================#
#code here
def endEffectorJacobianHW3(q:list[float])->list[float]:
    R,P,R_e,p_e = FKHW3(q)      #  FK for P_0_e - P_0_i to make a Geomatric Jacobian
    J = np.empty((6,3))
    for i in range(len(q)):
        J_v_i = np.cross(R[:,:,i][:,2],(p_e - P[:,i]))
        J_w_i = R[:,:,i][:,2]
        J_i = np.concatenate((J_v_i,J_w_i),axis=0)
        J[:,i] = J_i
    return J
#==============================================================================================================#
#=============================================<คำตอบข้อ 2>======================================================#
#code here
def checkSingularityHW3(q:list[float])->bool:
    threshold = 0.001
    J_full = endEffectorJacobianHW3(q)
    J_reduce = J_full[0:3,0:3]
    det = np.linalg.det(J_reduce)
    det_norm = np.abs(det)
    if det_norm < threshold:        # Singularity
        return True
    elif det_norm >= threshold:     # Not Singularity
        return False
#==============================================================================================================#
#=============================================<คำตอบข้อ 3>======================================================#
#code here
def computeEffortHW3(q:list[float], w:list[float])->list[float]:
    
    J = endEffectorJacobianHW3(q)
    R,P,R_e,p_e = FKHW3(q)
    
    f_e = w[3:]
    n_e = w[:3]
    
    f_0 = R_e @ f_e # Reframe to frame 0
    n_0 = R_e @ n_e
    
    w_0 = np.concatenate((f_0,n_0) , axis = 0)
    # print(w_0.shape)
    
    singularityFlag = checkSingularityHW3(q)
    if singularityFlag == True: # Singularity
        return 0
    elif singularityFlag == False:
        J_transpose = np.transpose(J)   # Transpose for calculate joint effort
        # print(J)
        # print(J_transpose)
        tau = J_transpose @ w_0
        # tau = J @ w_0
        return tau
#==============================================================================================================#
