#
# Resource Path
#
G1_XML_PATH = "/home/jiarui/Desktop/Resources/unitree_mujoco/g1/scene_29dof.xml"
G1_USD_PATH = "/home/jiarui/Desktop/Resources/unitree_isaac/G1/29dof/usd/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd"

LAFAN1_PATH = "/home/jiarui/Desktop/Dataset/LAFAN1/g1"
LAFAN1_INTERPOLATED_PATH = "/home/jiarui/Desktop/Dataset/LAFAN1/g1_interpolated"

LAFAN1_FPS = 30
LAFAN1_INTERPOLATED_FPS = 50

#
# Mujoco joint order
#
G1_MJ_JOINT_NAMES = [
    # pelvis position (3) + quaternion (4)
    # index [0:7]
    # "floating_base_joint",

    # left leg (6), index [7:13]
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", 
    "left_knee_joint", 
    "left_ankle_pitch_joint", "left_ankle_roll_joint",

    # right leg (6), index [13:19]
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", 
    "right_ankle_pitch_joint", "right_ankle_roll_joint",

    # waist (3), index [19:22]
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",

    # left arm (7), index [22:29]
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",

    # right arm (7), index [29:36]
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", 
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"
]


#
# Tracked Bodies
#
G1_ANCHOR_BODY_NAME = "torso_link"
G1_TRACKED_BODY_NAMES = [
    "torso_link",   # anchor body

    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link", # end-effector
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link", # end-effector
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link", # end-effector
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link" # end-effector
]

G1_END_EFFECTOR_BODY_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link"
]

G1_END_EFFECTOR_BODY_IDXES = [4, 7, 10, 13] # the index of end-effectors in tracked bodies

G1_MJ_ANCHOR_BODY_ID = 16
G1_MJ_TRACKED_BODY_IDS = [
    16, 
    1, 3, 5, 7, 9, 11, 13, 18, 20, 23, 25, 27, 30
]
G1_MJ_END_EFFECTOR_BODY_IDS = [7, 13, 23, 30]
G1_MJ_FEET_BODY_IDS = [7, 13]


G1_DEFAULT_JOINT_POS = {
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
}



# 
# PD parameters from BeyondMimic
#
ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ


G1_JOINT_PARAMS = {
    "left_hip_pitch_joint": {
        "STIFFNESS": STIFFNESS_7520_14,
        "DAMPING": DAMPING_7520_14,
        "ARMATURE": ARMATURE_7520_14,
        "TORQUE_LIM": 88.0,
        "VEL_LIM": 32.0,
    },
    "left_hip_roll_joint": {
        "STIFFNESS": STIFFNESS_7520_22,
        "DAMPING": DAMPING_7520_22,
        "ARMATURE": ARMATURE_7520_22,
        "TORQUE_LIM": 139.0,
        "VEL_LIM": 20.0,
    },
    "left_hip_yaw_joint": {
        "STIFFNESS": STIFFNESS_7520_14,
        "DAMPING": DAMPING_7520_14,
        "ARMATURE": ARMATURE_7520_14,
        "TORQUE_LIM": 88.0,
        "VEL_LIM": 32.0,
    },
    "left_knee_joint": {
        "STIFFNESS": STIFFNESS_7520_22,
        "DAMPING": DAMPING_7520_22,
        "ARMATURE": ARMATURE_7520_22,
        "TORQUE_LIM": 139.0,
        "VEL_LIM": 20.0,
    },
    "left_ankle_pitch_joint": {
        "STIFFNESS": 2.0 * STIFFNESS_5020,
        "DAMPING": 2.0 * DAMPING_5020,
        "ARMATURE": 2.0 * ARMATURE_5020,
        "TORQUE_LIM": 50.0,
        "VEL_LIM": 37.0,
    },
    "left_ankle_roll_joint": {
        "STIFFNESS": 2.0 * STIFFNESS_5020,
        "DAMPING": 2.0 * DAMPING_5020,
        "ARMATURE": 2.0 * ARMATURE_5020,
        "TORQUE_LIM": 50.0,
        "VEL_LIM": 37.0,
    },
    "right_hip_pitch_joint": {
        "STIFFNESS": STIFFNESS_7520_14,
        "DAMPING": DAMPING_7520_14,
        "ARMATURE": ARMATURE_7520_14,
        "TORQUE_LIM": 88.0,
        "VEL_LIM": 32.0,
    },
    "right_hip_roll_joint": {
        "STIFFNESS": STIFFNESS_7520_22,
        "DAMPING": DAMPING_7520_22,
        "ARMATURE": ARMATURE_7520_22,
        "TORQUE_LIM": 139.0,
        "VEL_LIM": 20.0,
    },
    "right_hip_yaw_joint": {
        "STIFFNESS": STIFFNESS_7520_14,
        "DAMPING": DAMPING_7520_14,
        "ARMATURE": ARMATURE_7520_14,
        "TORQUE_LIM": 88.0,
        "VEL_LIM": 32.0,
    },
    "right_knee_joint": {
        "STIFFNESS": STIFFNESS_7520_22,
        "DAMPING": DAMPING_7520_22,
        "ARMATURE": ARMATURE_7520_22,
        "TORQUE_LIM": 139.0,
        "VEL_LIM": 20.0,
    },
    "right_ankle_pitch_joint": {
        "STIFFNESS": 2.0 * STIFFNESS_5020,
        "DAMPING": 2.0 * DAMPING_5020,
        "ARMATURE": 2.0 * ARMATURE_5020,
        "TORQUE_LIM": 50.0,
        "VEL_LIM": 37.0,
    },
    "right_ankle_roll_joint": {
        "STIFFNESS": 2.0 * STIFFNESS_5020,
        "DAMPING": 2.0 * DAMPING_5020,
        "ARMATURE": 2.0 * ARMATURE_5020,
        "TORQUE_LIM": 50.0,
        "VEL_LIM": 37.0,
    },
    "waist_yaw_joint": {
        "STIFFNESS": STIFFNESS_7520_14,
        "DAMPING": DAMPING_7520_14,
        "ARMATURE": ARMATURE_7520_14,
        "TORQUE_LIM": 88.0,
        "VEL_LIM": 32.0,
    },
    "waist_roll_joint": {
        "STIFFNESS": 2.0 * STIFFNESS_5020,
        "DAMPING": 2.0 * DAMPING_5020,
        "ARMATURE": 2.0 * ARMATURE_5020,
        "TORQUE_LIM": 50.0,
        "VEL_LIM": 37.0,
    },
    "waist_pitch_joint": {
        "STIFFNESS": 2.0 * STIFFNESS_5020,
        "DAMPING": 2.0 * DAMPING_5020,
        "ARMATURE": 2.0 * ARMATURE_5020,
        "TORQUE_LIM": 50.0,
        "VEL_LIM": 37.0,
    },
    "left_shoulder_pitch_joint": {
        "STIFFNESS": STIFFNESS_5020,
        "DAMPING": DAMPING_5020,
        "ARMATURE": ARMATURE_5020,
        "TORQUE_LIM": 25.0,
        "VEL_LIM": 37.0,
    },
    "left_shoulder_roll_joint": {
        "STIFFNESS": STIFFNESS_5020,
        "DAMPING": DAMPING_5020,
        "ARMATURE": ARMATURE_5020,
        "TORQUE_LIM": 25.0,
        "VEL_LIM": 37.0,
    },
    "left_shoulder_yaw_joint": {
        "STIFFNESS": STIFFNESS_5020,
        "DAMPING": DAMPING_5020,
        "ARMATURE": ARMATURE_5020,
        "TORQUE_LIM": 25.0,
        "VEL_LIM": 37.0,
    },
    "left_elbow_joint": {
        "STIFFNESS": STIFFNESS_5020,
        "DAMPING": DAMPING_5020,
        "ARMATURE": ARMATURE_5020,
        "TORQUE_LIM": 25.0,
        "VEL_LIM": 37.0,
    },
    "left_wrist_roll_joint": {
        "STIFFNESS": STIFFNESS_5020,
        "DAMPING": DAMPING_5020,
        "ARMATURE": ARMATURE_5020,
        "TORQUE_LIM": 25.0,
        "VEL_LIM": 37.0,
    },
    "left_wrist_pitch_joint": {
        "STIFFNESS": STIFFNESS_4010,
        "DAMPING": DAMPING_4010,
        "ARMATURE": ARMATURE_4010,
        "TORQUE_LIM": 5.0,
        "VEL_LIM": 22.0,
    },
    "left_wrist_yaw_joint": {
        "STIFFNESS": STIFFNESS_4010,
        "DAMPING": DAMPING_4010,
        "ARMATURE": ARMATURE_4010,
        "TORQUE_LIM": 5.0,
        "VEL_LIM": 22.0,
    },
    "right_shoulder_pitch_joint": {
        "STIFFNESS": STIFFNESS_5020,
        "DAMPING": DAMPING_5020,
        "ARMATURE": ARMATURE_5020,
        "TORQUE_LIM": 25.0,
        "VEL_LIM": 37.0,
    },
    "right_shoulder_roll_joint": {
        "STIFFNESS": STIFFNESS_5020,
        "DAMPING": DAMPING_5020,
        "ARMATURE": ARMATURE_5020,
        "TORQUE_LIM": 25.0,
        "VEL_LIM": 37.0,
    },
    "right_shoulder_yaw_joint": {
        "STIFFNESS": STIFFNESS_5020,
        "DAMPING": DAMPING_5020,
        "ARMATURE": ARMATURE_5020,
        "TORQUE_LIM": 25.0,
        "VEL_LIM": 37.0,
    },
    "right_elbow_joint": {
        "STIFFNESS": STIFFNESS_5020,
        "DAMPING": DAMPING_5020,
        "ARMATURE": ARMATURE_5020,
        "TORQUE_LIM": 25.0,
        "VEL_LIM": 37.0,
    },
    "right_wrist_roll_joint": {
        "STIFFNESS": STIFFNESS_5020,
        "DAMPING": DAMPING_5020,
        "ARMATURE": ARMATURE_5020,
        "TORQUE_LIM": 25.0,
        "VEL_LIM": 37.0,
    },
    "right_wrist_pitch_joint": {
        "STIFFNESS": STIFFNESS_4010,
        "DAMPING": DAMPING_4010,
        "ARMATURE": ARMATURE_4010,
        "TORQUE_LIM": 5.0,
        "VEL_LIM": 22.0,
    },
    "right_wrist_yaw_joint": {
        "STIFFNESS": STIFFNESS_4010,
        "DAMPING": DAMPING_4010,
        "ARMATURE": ARMATURE_4010,
        "TORQUE_LIM": 5.0,
        "VEL_LIM": 22.0,
    },
}



G1_ACTION_SCALE = {
    joint_name: 0.25 * params["TORQUE_LIM"] / params["STIFFNESS"]
    for joint_name, params in G1_JOINT_PARAMS.items()
}

if __name__ == '__main__':
    
    """
    Get the body ids in TrackingConfig. Only run once
    """
    import mujoco
    model = mujoco.MjModel.from_xml_path(G1_XML_PATH)
    data = mujoco.MjData(model)


    # print("--------------- Tracked Bodies ---------------")
    # for body_name in G1_TRACKED_BODY_NAMES:
    #     body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    #     print(body_name, body_id)

    print("--------------- End-Effectors ---------------")
    for body_name in G1_END_EFFECTOR_BODY_NAMES:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        print(body_name, body_id)