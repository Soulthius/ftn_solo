import numpy as np
from ftn_solo.utils.pinocchio import PinocchioWrapper
import time


def float_or_list(value, num_joints):
    return np.array(value if type(value) is list else [float(value)]*num_joints, dtype=np.float64)


class RneAlgorithm(PinocchioWrapper):
    def __init__(self, num_joints, yaml_config, robot_version, logger, dt) -> None:
        super().__init__(robot_version, logger, dt)
        # self.Kp = float_or_list(yaml_config["Kp"], num_joints)
        # self.Kd = float_or_list(yaml_config["Kd"], num_joints)
        self.B = float_or_list(yaml_config["B"], num_joints)
        self.Fv = float_or_list(yaml_config["Fv"], num_joints)
        # self.max_control = float_or_list(
        #     yaml_config["max_control"], num_joints)
        self.logger = logger
        self.q = np.zeros(19)
        self.dq_curr = np.array([])
        self.dq = np.array([])
        self.nddq = np.zeros(18)
        self.J_real =np.zeros((3, 18))
        self.q_base = np.array([0,0,0,0,0,0,1])
        self.dq_base = np.array([0,0,0,0,0,0])
        self.tau_con = np.zeros(12)
        self.Kp=0
        self.Kd=0
       
    def compute_kinematics(self,qcurr, dqcurr):
        # qbase = np.array([qbase[1],qbase[2],qbase[3],qbase[0]])
        # self.q = np.concatenate((np.concatenate((self.q_base,qbase)), qcurr))
        self.q = np.concatenate((self.q_base,qcurr))
        self.dq_curr = np.concatenate((self.dq_base,dqcurr))
        self.J_real.fill(0)
        self.frame_forward_kinematics(self.q, self.dq_curr, self.end_eff_ids)
        self.compute_frame_jacobian(self.q,self.dq_curr)
        self.compute_non_linear(self.q, self.dq_curr) 

    def compute_acceleration(self,frame_id,poz,vel,acc):

        current_poz = self.data.oMf[frame_id].translation
        current_vel = self.get_frame_velocity(frame_id)
        J_real,J_dot = self.get_frame_jacobian(frame_id)
        
        ref_acc = acc + self.Kp*(poz - current_poz) + self.Kd*(vel - current_vel[:3])

        ddq = np.dot(np.linalg.pinv(J_real),ref_acc - J_dot[:3])
        
        return ddq
        
        

        
      
    
    def get_tourque(self,ddq):

        self.tau = self.compute_recrusive_newtone_euler(self.ndq, ddq,self.Fv,self.B)
        return self.tau 
 