import pinocchio as pin
import numpy as np
from robot_properties_solo.solo12wrapper import Solo12Config


class PinocchioWrapper(object):
    def __init__(self, robot_version, logger, dt):

        # self.resourses = Resources(robot_version)
        self.pin_robot = Solo12Config.buildRobotWrapper()
        self.model = self.pin_robot.model
        self.data = self.model.createData()
        self.logger = logger
        self.dt = dt
        self.prev_err = np.inf
        self.delta_error = np.inf
        self.nu = []
        self.M = np.array([])
        self.C = np.array([])
        self.G = np.array([])

        self.end_eff_ids = []
        self.end_effector_names = []
        controlled_joints = []

        for leg in ["FL", "FR", "HL", "HR"]:
            controlled_joints += [leg + "_HAA", leg + "_HFE", leg + "_KFE"]
            self.end_eff_ids.append(
                self.model.getFrameId(leg + "_ANKLE")
            )
            self.end_effector_names.append(leg + "_ANKLE")

        self.base_link = self.model.getFrameId("base_link")

        self.fr = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED

        # Parameters for IK
        self.DT = 0.01
        self.epsilon = 0.00000001
        self.k_max = 0.00000002
        self.J = np.zeros((6, 18))
        self.J_list=[]
        self.max_tau = 10.0

    def mass(self, q):
        return pin.crba(self.model, self.data, q)

    def gravity(self, q):
        return pin.computeGeneralizedGravity(self.model, self.data, q)

    def moveSE3(self, R, T):
        oMdes = pin.SE3(R, T)
        return oMdes

    def pinIntegrate(self, q, v):
        return pin.integrate(self.model, q, v*self.DT)

    

    def frame_forward_kinematics(self, q, dq):
        pin.framesForwardKinematics(self.model, self.data, q)
        pin.forwardKinematics(self.model, self.data,q,dq,0*dq)
        

    def compute_frame_jacobian(self, q,dq):
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

    def compute_non_linear(self, q, dq):
        self.M = pin.crba(self.model, self.data, q)
        self.C = pin.computeCoriolisMatrix(self.model, self.data, q, dq)
        self.G = pin.computeGeneralizedGravity(self.model, self.data, q)
        
    
    
    def get_frame_jacobian(self, frame_id):
        self.J.fill(0)
        J_dot = np.zeros(18)
        self.J = pin.getFrameJacobian(self.model,self.data,frame_id,self.fr)
        J_dot = pin.getFrameJacobianTimeVariation(self.model, self.data,frame_id,self.fr)
        
        
        self.J[:, :6] = 0
      
     
        return  self.J[:3,:], J_dot[:3,6:]
    
    def get_frame_velocity(self,frame_id):
        return pin.getFrameVelocity(self.model, self.data,frame_id,self.fr)
    


    def pd_controller(self, ref_pos, ref_vel, position, velocity,t,kp):
        if t < 4:
            Kp = kp + t * 300000 
            Kd = 200
        else:
            Kp = kp
            Kd = 200
        
        
       
        pos_diff = ref_pos - position
        vel_diff = ref_vel - velocity
        # self.logger.info("pos_diff: {}".format(pos_diff))

        control =  Kp * (pos_diff) + Kd * (vel_diff)
        # return np.concatenate((np.zeros(6),control))
        return control
    
   

    def compute_recrusive_newtone_euler(self, dq, ddq,Fv,B):


        #Projection matrix 
        # J_M_inv = np.dot(J,np.linalg.inv(self.M[6:, 6:]))

        # lambda_matrix = np.linalg.pinv(np.dot(J_M_inv, J.T))
        # P = np.eye(12) - np.dot(np.dot(J.T, lambda_matrix), J_M_inv)
     
        # h=np.dot(self.C[6:, 6:], dq[6:]) +  self.G[6:]
        # p_ddq = np.dot(P, ddq) 

        # tau = np.dot(P, np.dot(self.M[6:, 6:], ddq) + h)


        # Augmented system 
        # h=np.dot(self.C[6:, 6:], dq[6:]) +  self.G[6:]
        # zero_block = np.zeros((J.shape[0], J.shape[0]))
        # A_aug = np.block([
        #     [self.M[6:, 6:], J.T],
        #     [J, zero_block]
        #     ])
        
        # dynamics_rhs = np.dot(self.M[6:, 6:], ddq) + h
        # constraint_rhs = -np.dot(J_dot, dq[6:])

        # b_aug = np.concatenate([dynamics_rhs, constraint_rhs])

        # solution = np.linalg.lstsq(A_aug, b_aug, rcond=1e-6)[0]
        # ddq_test = solution[:12]         # Joint accelerations
        # lambda_vector = solution[12:]  # Constraint forces

        # tau = np.dot(self.M[6:, 6:], ddq_test) + h + np.dot(J.T, lambda_vector) + np.dot(Fv,dq[6:]) + B   # Add constraint contribution



# 
        tau = np.dot(self.M[6:, 6:], ddq[6:]) + np.dot(self.C[6:, 6:], dq[6:]) + np.dot(Fv,dq[6:]) + B + self.G[6:]

        self.logger.info("tau_test: {}".format(tau))

        # return np.clip(tau, -self.max_tau, self.max_tau)
        # return tau[6:]
        return tau