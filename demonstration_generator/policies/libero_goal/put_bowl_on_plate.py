import numpy as np
from robosuite.utils.transform_utils import (
    mat2quat,
    quat_conjugate,
    quat_multiply,
    quat2axisangle,
)


# ---------------------------------------------------------------------------
# Smooth movement helpers
# ---------------------------------------------------------------------------

def nlerp(q1, q2, alpha):
    """Normalised linear interpolation for smooth rotations."""
    if np.dot(q1, q2) < 0:
        q2 = -q2

    q_interp = (1 - alpha) * q1 + alpha * q2
    return q_interp / np.linalg.norm(q_interp)


def get_matrix(env, name):
    """Get the world-frame pose matrix of an object/body."""
    pos = env.sim.data.get_body_xpos(name).copy()
    mat = env.sim.data.get_body_xmat(name).reshape(3, 3).copy()

    transform = np.eye(4)
    transform[:3, :3] = mat
    transform[:3, 3] = pos

    return transform


def move_to_smooth(
    env,
    target_matrix,
    offset=None,
    steps=100,
    grip=-1.0,
    pos_gain=3.0,
    rot_gain=2.0,
):
    """Smoothly move the end-effector to a target pose."""
    if offset is None:
        offset = [0, 0, 0]

    target_pos_world = target_matrix[:3, 3] + np.array(offset)
    target_quat = mat2quat(target_matrix[:3, :3])

    robot_base_pos = np.array(env.robots[0].base_pos)

    start_pos_world = env.robots[0]._hand_pos + robot_base_pos
    start_quat = env.robots[0]._hand_quat.copy()

    for step in range(1, steps + 1):
        alpha = step / steps

        interp_pos_world = start_pos_world + alpha * (
            target_pos_world - start_pos_world
        )
        interp_quat = nlerp(start_quat, target_quat, alpha)

        curr_pos_world = env.robots[0]._hand_pos + robot_base_pos
        curr_quat = env.robots[0]._hand_quat

        pos_error = interp_pos_world - curr_pos_world

        if np.dot(curr_quat, interp_quat) < 0:
            interp_quat = -interp_quat

        rel_quat = quat_multiply(interp_quat, quat_conjugate(curr_quat))
        rot_error = quat2axisangle(rel_quat)

        action = np.zeros(7)
        action[:3] = pos_error * pos_gain
        action[3:6] = rot_error * rot_gain
        action[6] = grip

        env.step(action)

        if getattr(env, "has_renderer", False):
            env.render()


def gripper_action(env, cmd, steps=40):
    """Open or close the gripper for a fixed number of steps."""
    action = np.zeros(7)
    action[6] = cmd

    for _ in range(steps):
        env.step(action)

        if getattr(env, "has_renderer", False):
            env.render()


# ---------------------------------------------------------------------------
# Captured body-to-body transforms
# ---------------------------------------------------------------------------

# Parent = akita_black_bowl_1_main, Child = robot0_right_hand
T_HAND_ON_BOWL = np.array(
    [
        [0.996954, -0.02150385, -0.07496874, 0.0087228],
        [-0.02373346, -0.99929827, -0.02897746, 0.05306893],
        [-0.074293, 0.03066846, -0.99676476, 0.03358268],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

# Parent = plate_1_main, Child = akita_black_bowl_1_main
T_BOWL_ON_PLATE = np.array(
    [
        [0.99234336, -0.12171498, 0.02097882, 0.00275265],
        [0.12132078, 0.99242876, 0.01914206, -0.0119236],
        [-0.02314986, -0.01645033, 0.99959665, 0.00515369],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


# ---------------------------------------------------------------------------
# Main policy
# ---------------------------------------------------------------------------

def run_solver(env, bddl_file=None):
    """
    Scripted policy for:
    put_the_bowl_on_the_plate

    The environment is created by run_collection.py.
    This function only controls the robot.
    """
    print("Phase 1: Picking up bowl...")

    bowl_start = get_matrix(env, "akita_black_bowl_1_main") @ T_HAND_ON_BOWL

    move_to_smooth(env, bowl_start, offset=[0, 0, 0.12], steps=100, grip=-1.0)
    move_to_smooth(env, bowl_start, offset=[0, 0, 0.00], steps=100, grip=-1.0)

    gripper_action(env, cmd=1.0, steps=40)

    move_to_smooth(env, bowl_start, offset=[0, 0, 0.15], steps=100, grip=1.0)

    print("Phase 2: Placing bowl on plate...")

    goal_pose = get_matrix(env, "plate_1_main") @ T_BOWL_ON_PLATE @ T_HAND_ON_BOWL

    move_to_smooth(env, goal_pose, offset=[0, 0, 0.10], steps=100, grip=1.0)
    move_to_smooth(env, goal_pose, offset=[0, 0, 0.02], steps=100, grip=1.0)

    gripper_action(env, cmd=-1.0, steps=40)

    move_to_smooth(env, goal_pose, offset=[0, 0, 0.20], steps=100, grip=-1.0)

    print("Mission complete.")

    return True