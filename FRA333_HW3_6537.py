# file สำหรับเขียนคำตอบ
# ในกรณีที่มีการสร้าง function อื่น ๆ ให้ระบุว่า input-output คืออะไรด้วย
from HW3_utils import FKHW3
import numpy as np
# '''
# ชื่อ_รหัส(ธนวัฒน์_6461)
# 1. ปวริศ_6537
# 2.
# 3.
# '''
#=============================================<คำตอบข้อ 1>======================================================#
#code here
def endEffectorJacobianHW3(q:list[float])->list[float]:
    R,P,R_e,p_e = FKHW3(q)      #   FK for P_0_e - P_0_i to make a Geomatric Jacobian
    J = np.empty((6,3))         #   Create Empty Matrix   
    for i in range(len(q)):
        J_v_i = np.cross(R[:,:,i][:,2],(p_e - P[:,i]))      # for linear velocity   Z cross (P_e - P_i)
        J_w_i = R[:,:,i][:,2]                               # for angular velocity  Z
        J_i = np.concatenate((J_v_i,J_w_i),axis=0)          # Concatenate the jacobian component
        J[:,i] = J_i            #   Append Jacobian of each joint
    return J
#==============================================================================================================#
#=============================================<คำตอบข้อ 2>======================================================#
#code here
def checkSingularityHW3(q:list[float])->bool:
    threshold = 0.001                       #   Define Threshold
    J_full = endEffectorJacobianHW3(q)      #   Call function to calculate frame 0
    J_reduce = J_full[0:3,0:3]              #   Reduce Jacobian to 3 DoF (x,y,z)
    det = np.linalg.det(J_reduce)           #   det of Jacobian
    det_norm = np.abs(det)                  #   norm of Jacobian to check singu
    if det_norm < threshold:        # Singularity
        return True
    elif det_norm >= threshold:     # Not Singularity
        return False
#==============================================================================================================#
#=============================================<คำตอบข้อ 3>======================================================#
#code here
def computeEffortHW3(q:list[float], w:list[float])->list[float]:
    J = endEffectorJacobianHW3(q)       #   Calculate Jacobian
    R,P,R_e,p_e = FKHW3(q)              #   Get Component from FK
    f_e = w[3:]         #   Force from wrench
    n_e = w[:3]         #   Moment from wrench 
    f_0 = R_e @ f_e     # Reframe to frame 0
    n_0 = R_e @ n_e
    w_0 = np.concatenate((f_0,n_0) , axis = 0)      #   Concatenate matrix to [f_0 ; n_0]
    singularityFlag = checkSingularityHW3(q)        #   Check Singularity
    if singularityFlag == True:     # is Singularity
        return 0
    elif singularityFlag == False:  # is not Singularity
        J_transpose = np.transpose(J)   # Transpose for calculate joint effort
        tau = J_transpose @ w_0         # Calculate torque that effect to 
        return tau
#==============================================================================================================#
