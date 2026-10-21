import os

import numpy as np
import mujoco
import mujoco.viewer
from scipy.io import loadmat, savemat
from scipy.spatial.transform import Rotation as R
import time
from tqdm import tqdm

import sys
sys.path.append("..")
import config

def visualize_frame(q_hist, fps, quaternion_convention="wxyz"):
    """
    :param q_hist: np.array, [num_frame, 36]
    :param fps: int
    """

    dt = 1.0 / fps
    model = mujoco.MjModel.from_xml_path(config.G1_XML_PATH)
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for frame in q_hist:
            # Avoid accessing native viewer resources after the window closes.
            if not viewer.is_running():
                break

            quat = frame[3:7]

            # The passive viewer runs on another thread. Protect shared
            # model, data, and camera state while modifying them.
            with viewer.lock():
                data.qpos[0:3] = frame[0:3]

                if quaternion_convention == "wxyz":
                    data.qpos[3:7] = quat
                elif quaternion_convention == "xyzw":
                    data.qpos[3:7] = [quat[3], quat[0], quat[1], quat[2]]

                data.qpos[7:] = frame[7:]
                # mujoco.mj_forward(model, data)
                viewer.cam.lookat[:] = data.qpos[0:3]

            # Synchronize after releasing the explicit lock.
            viewer.sync(state_only=True)
            time.sleep(dt)


def interpolate_frames(input_q_hist, input_fps:int, output_fps:int):
    """
    Interpolate a motion clip to a higher fps.
    Note that input_frames (LAFAN1 dataset) use [x,y,z,w] convention for quaternion.
    But the output use [w,x,y,z] convention for quaternion
    
    :param input_q_hist: np.array, [num_input_frames, 36]
    :param input_fps: int
    :param output_fps: int
    :return output_q_hist: np.array, [num_output_frames, 36]
    :return output_qdot_hist: np.array, [num_output_frames, 35]
    """

    input_dt = 1.0 / input_fps
    output_dt = 1.0 / output_fps

    duration = (len(input_q_hist) - 1) * input_dt
    input_ts = np.arange(0, duration, input_dt)
    output_ts = np.arange(0, duration, output_dt)

    output_q_hist = []
    output_qdot_hist = []

    for i in range(0, len(output_ts)):
        
        # find the neighbor frames
        phase = output_ts[i] / duration + 1e-7
        index_0 = np.floor( (phase * (len(input_q_hist)-1)) ).astype(np.int64)
        index_1 = min( index_0 + 1, len(input_q_hist) - 1 )
        dt = output_ts[i] - input_ts[index_0]

        # extract root position, root quaternion and joint angles
        # the data use quaternion convention [x,y,z,w], but we use [w, x, y, z]
        frame_0 = input_q_hist[index_0]
        frame_1 = input_q_hist[index_1]
        p0 = frame_0[:3]; p1 = frame_1[:3]
        R0 = R.from_quat(frame_0[3:7]); R1 = R.from_quat(frame_1[3:7])
        j0 = frame_0[7:]; j1 = frame_1[7:]

        # compute linear velocitys of root and joint angles
        # then linear interpolate
        p_vel = (p1 - p0) / input_dt; p = p0 + p_vel * dt
        j_vel = (j1 - j0) / input_dt; j = j0 + j_vel * dt

        # compute angular velocity (body frame) of root and interpolate using quaternion
        R_delta = R0.inv() * R1
        rotvec = R_delta.as_rotvec()
        angular_vel = rotvec / input_dt
        R_interp = R0 * R.from_rotvec(angular_vel * dt)
        q = R_interp.as_quat(scalar_first=True)

        # record the interpolation and velocity
        output_q_hist.append(
            np.concatenate([p, q, j])
        )
        output_qdot_hist.append(
            np.concatenate([p_vel, angular_vel, j_vel])
        )
        

    return np.stack(output_q_hist), np.stack(output_qdot_hist)

def extract_from_mujoco(model, data):
    """
    Extract the body poses of the frame in data

    :return poses: np.array, [num_tracked_body, 7]
        Every 7 elements express the position and quaternion (in world frame) of a body

    :return vels: np.array, [num_tracked_body, 6]
        Every 6 elements express the linear velocity (in world frame) and angular velocity (in world frame) of a body
    """
    poses = []
    vels = []

    vel_holder = np.zeros([6])

    # poses for bodies
    for body_id in config.G1_MJ_TRACKED_BODY_IDS:
        body_pos = data.xpos[body_id].copy()
        body_quat = data.xquat[body_id].copy()

        mujoco.mj_objectVelocity(
            model,
            data,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
            vel_holder,
            0, # world-frame velocity
        )
        poses.append(np.concatenate([body_pos, body_quat]))
        vels.append(np.concatenate([vel_holder[3:6], vel_holder[0:3]])) # [0:3] is angular velocity, [3:6] is linear velocity

    poses= np.stack(poses)
    vels = np.stack(vels)
    return poses, vels


def extract_from_frames(q_hist, qdot_hist):
    """
    For each frame, extract the position (world frame), quaternion (world frame), linear velocity (world frame), angular velocity (world frame) for each tracked body
    This will be used later for computing anchor transform and rewards 
    
    :param q_hist: np.array, [num_frames, 36]
    :param qdot_hist: np.array, [num_frames, 35]

    :return poses: np.array, [num_frames, num_tracked_body, 7]
    :return vels: np.array, [num_frames, num_tracked_body, 6]
    """

    model = mujoco.MjModel.from_xml_path(config.G1_XML_PATH)
    data = mujoco.MjData(model)

    poses_hist = []
    vels_hist = []

    for qpos, qvel in zip(q_hist, qdot_hist):

        data.qpos[:] = qpos
        data.qvel[:] = qvel

        # compute forward kinematics using mujoco
        mujoco.mj_forward(model, data)

        poses, vels = extract_from_mujoco(model, data)
        poses_hist.append(poses)
        vels_hist.append(vels)

    return np.stack(poses_hist), np.stack(vels_hist)



if __name__ == '__main__':

    """
    visualize an original motion clip
    """
    # motion_file = os.path.join(config.LAFAN1_PATH, "walk1_subject1.csv")
    # q_hist = np.genfromtxt(motion_file, delimiter=",")[:200]
    # print(q_hist.shape)
    # visualize_frame(q_hist, fps=config.LAFAN1_FPS, quaternion_convention="xyzw")


    """
    visualize an interpolated motion clip
    """
    # motion_file = os.path.join(config.LAFAN1_PATH, "walk1_subject1.csv")
    # q_hist = np.genfromtxt(motion_file, delimiter=",")[:200]
    # interpolated_q_hist, interpolated_qdot_hist = interpolate_frames(q_hist, config.LAFAN1_FPS, config.LAFAN1_INTERPOLATED_FPS)
    # print(interpolated_q_hist.shape, interpolated_qdot_hist.shape)
    # visualize_frame(interpolated_q_hist, fps=config.LAFAN1_INTERPOLATED_FPS, quaternion_convention="wxyz")


    """
    Interpolate all motions
    And precompute the pose and velocity of each tracked body
    """
    # os.makedirs(config.LAFAN1_INTERPOLATED_PATH, exist_ok=True)
    # folder = config.LAFAN1_PATH
    # for name in tqdm(os.listdir(folder), desc="Interpolating motions"):
    #     path = os.path.join(folder, name)
    #     if os.path.isfile(path):

    #         input_q_hist = np.genfromtxt(path, delimiter=",")
    #         interpolated_q_hist, interpolated_qdot_hist = interpolate_frames(input_q_hist, config.LAFAN1_FPS, config.LAFAN1_INTERPOLATED_FPS)

    #         interpolated_poses_hist, interpolated_vels_hist = extract_from_frames(interpolated_q_hist, interpolated_qdot_hist)
    
    #         savemat(
    #             os.path.join(config.LAFAN1_INTERPOLATED_PATH, name)[:-4] + ".mat",
    #             {
    #                 "q_hist":interpolated_q_hist,
    #                 "qdot_hist":interpolated_qdot_hist,
    #                 "poses_hist":interpolated_poses_hist,
    #                 "vels_hist":interpolated_vels_hist
    #             }
    #         )

    #         print( os.path.join(config.LAFAN1_INTERPOLATED_PATH, name)[:-4] + ".mat" )

    """
    Visualization and Sanity Check
    q_hist: [num_frames, 7+29]
    qdot_hist: [num_frames, 6+29]
    poses_hist: [num_frames, num_tracked_bodies, 7]
    vels_hist: [num_frames, num_tracked_bodies, 6]
    """
    motion = "walk1_subject2"
    motion = "fallAndGetUp1_subject1"
    interpolated = True
    max_len = 4000

    if interpolated:
        path = os.path.join(config.LAFAN1_INTERPOLATED_PATH, motion+".mat")
        mat = loadmat(path)
        q_hist, qdot_hist, poses_hist, vels_hist = mat["q_hist"], mat["qdot_hist"], mat["poses_hist"], mat["vels_hist"]
        print(q_hist.shape, qdot_hist.shape, poses_hist.shape, vels_hist.shape)
        fps = config.LAFAN1_INTERPOLATED_FPS
        quaternion_convention="wxyz"

    else:
        path = os.path.join(config.LAFAN1_PATH, motion+".csv")
        q_hist = np.genfromtxt(path, delimiter=",")
        fps = config.LAFAN1_FPS
        quaternion_convention="xyzw"

    # gif = visualize_frame(q_hist[:max_len], fps, quaternion_convention)
    print(q_hist[0])
