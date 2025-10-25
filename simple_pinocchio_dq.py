import numpy as np
import pinocchio as pin
from robot_properties_solo.config import Solo12Config

robot = Solo12Config.buildRobotWrapper()
model = robot.model

data = model.createData()



pin.forwardKinematics(model, data, q, dq)
pin.updateFramePlacements(model, data)
pin.computeJointJacobians(model, data, q)

foot_frame_id = model.getFrameId("foot")

pos, vel,acc = get_trajectory()
vel_frame = pin.getFrameVelocity(model, data, foot_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
vel_diff = vel - vel_frame.linear


J = pin.getFrameJacobian(model, data, foot_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
J_linear = J[:3, :]

dq = np.dot(np.linalg.pinv(J_linear),vel_diff)
