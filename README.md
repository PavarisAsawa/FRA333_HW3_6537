# FRA333 Homework Assignment 3: Static Force
## Objective :
This assignment is designed to allow students to apply their knowledge of differential kinematics for a 3-degree-of-freedom (3-DOF) robotic manipulator.

## RRR Robot

![alt text](image.png)

## Question 1
Find a Jacobian matrix at end-effector frame relate to base frame

```py
  def endEffectorJacobianHW3(q:list[float])->list[float]:
    R,P,R_e,p_e = FKHW3(q)      #   FK for P_0_e - P_0_i to make a Geomatric Jacobian
    J = np.empty((6,3))         #   Create Empty Matrix   
    for i in range(len(q)):
        J_v_i = np.cross(R[:,:,i][:,2],(p_e - P[:,i]))      # for linear velocity   Z cross (P_e - P_i)
        J_w_i = R[:,:,i][:,2]                               # for angular velocity  Z
        J_i = np.concatenate((J_v_i,J_w_i),axis=0)          # Concatenate the jacobian component
        J[:,i] = J_i            #   Append Jacobian of each joint
    return J
```