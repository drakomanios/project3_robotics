import pinocchio as pin
import numpy as np


def urdf_extract(urdf_file):
    model = pin.buildModelFromUrdf(urdf_file)
    data = model.createData()
    return model, data

def urdf_fk_frame(model, data, q = None, frame_name = None, Tws = None, apply_qlim = True):
    if q is None: q = np.zeros(model.nq)
    if apply_qlim: q = np.minimum(np.maximum(q, model.lowerPositionLimit), model.upperPositionLimit)
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    if frame_name is None: frame_name = list(model.frames)[-1].name
    frame_id = model.getFrameId(frame_name)
    frame_pose = data.oMf[frame_id]
    if Tws is not None: frame_pose = Tws @ frame_pose
    return frame_pose

def urdf_fk_joint(model, data, q = None, joint_name = None, Tws = None, apply_qlim = True):
    if q is None: q = np.zeros(model.nq)
    if apply_qlim: q = np.minimum(np.maximum(q, model.lowerPositionLimit), model.upperPositionLimit)
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    if joint_name is None: joint_name = list(model.names)[-1]
    joint_id = model.getJointId(joint_name)
    joint_pose = data.oMi[joint_id]
    if Tws is not None: frame_pose = Tws @ frame_pose
    return joint_pose
