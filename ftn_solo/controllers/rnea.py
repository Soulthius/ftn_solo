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
        self.Kp=3000000
        self.Kd=200
       
    def compute_kinematics(self,qcurr, dqcurr):
        # qbase = np.array([qbase[1],qbase[2],qbase[3],qbase[0]])
        # self.q = np.concatenate((np.concatenate((self.q_base,qbase)), qcurr))
        self.q = np.concatenate((self.q_base,qcurr))
        self.dq_curr = np.concatenate((self.dq_base,dqcurr))
        self.J_real.fill(0)
        self.frame_forward_kinematics(self.q, self.dq_curr)
        self.compute_frame_jacobian(self.q,self.dq_curr)
        self.compute_non_linear(self.q, self.dq_curr) 

    def compute_acceleration(self,joint_id,poz,vel,acc):
        frame_id = self.model.getFrameId(joint_id + "_ANKLE")
        current_poz = self.data.oMf[frame_id]
        current_vel = self.get_frame_velocity(frame_id)
        J_real,J_dot = self.get_frame_jacobian(frame_id)

        self.logger.info("Goal poz:{}".format(poz))

        iMd = current_poz.actInv(poz)
            
        pos_diff= self.pin_log(iMd)
        
        # pos_diff = poz - current_poz
        # vel_diff = vel - current_vel.linear

        # ref_acc =  self.Kp*pos_diff + self.Kd*vel_diff
        # acc_diff = ref_acc - J_dot
        dq = np.dot(np.linalg.pinv(J_real),pos_diff[:3])
        # ref_ddq =  self.Kp*(self.q[7:] - q[7:]) + self.Kd*(self.dq_curr[6:]-dq[6:])
        # ddq = np.dot(np.linalg.pinv(J_real),acc_diff)


        # self.logger.info("ref_acc: {}".format(ref_ddq))
        
        # self.logger.info("acc_diff: {}".format(acc_diff))
        # self.logger.info("ddq: {}".format(ddq))
        # self.logger.info("acc:{}".format(acc))
        
        return dq
        
        

    def get_acceleration(self,dq):
        q=self.pinIntegrate(self.q,dq)
        return self.Kp*(self.q[7:] - q[7:]) + self.Kd*(self.dq_curr[6:]-dq[6:]) 
      
    
    def get_tourque(self,dq,ddq):

        self.tau = self.compute_recrusive_newtone_euler(dq,ddq,self.Fv,self.B)

        return self.tau 
 