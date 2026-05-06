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
    pos_gain=4.0,
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

# Parent = porcelain_mug_1_main, Child = robot0_right_hand
T_HAND_ON_WHITE_MUG = np.array(
    [
        [0.99917221, -0.01960553, -0.03564423, -0.05372994],
        [-0.02025307, -0.99963468, -0.01789732, 0.00160802],
        [-0.03528032, 0.01860441, -0.99920427, 0.11797341],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

# Parent = plate_1_main, Child = porcelain_mug_1_main
T_WHITE_MUG_ON_PLATE = np.array(
    [
        [-9.97736873e-01, -6.56546571e-02, 1.45120108e-02, 0.01480494],
        [6.53321550e-02, -9.97628022e-01, -2.16803665e-02, -0.00076136],
        [1.59010056e-02, -2.06832001e-02, 9.99659624e-01, 0.00393180],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

# Parent = white_yellow_mug_1_main, Child = robot0_right_hand
T_HAND_ON_YELLOW_MUG = np.array(
    [
        [0.99396924, -0.10235538, -0.03935138, 0.06588267],
        [-0.10203466, -0.99472978, 0.01007919, -0.00074188],
        [-0.04017565, -0.00600320, -0.99917460, 0.12048087],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

# Parent = plate_2_main, Child = white_yellow_mug_1_main
T_YELLOW_MUG_ON_PLATE = np.array(
    [
        [-0.99952579, -0.01987753, -0.02351752, -0.01547564],
        [0.02026391, -0.99966167, -0.0163069, 0.00516768],
        [-0.02318543, -0.01677572, 0.99959042, 0.00557832],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


# ---------------------------------------------------------------------------
# Main policy
# ---------------------------------------------------------------------------

def run_solver(env, bddl_file=None):
    """
    Scripted policy for:
    put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate

    The environment is created by run_collection.py.
    This function only controls the robot.
    """
    print("Phase 1: Placing white mug on left plate...")

    white_pick = get_matrix(env, "porcelain_mug_1_main") @ T_HAND_ON_WHITE_MUG

    move_to_smooth(env, white_pick, offset=[0, 0, 0.15], steps=100, grip=-1.0)
    move_to_smooth(env, white_pick, offset=[0, 0, 0.00], steps=100, grip=-1.0)

    gripper_action(env, cmd=1.0, steps=40)

    move_to_smooth(env, white_pick, offset=[0, 0, 0.20], steps=100, grip=1.0)

    white_goal = get_matrix(env, "plate_1_main") @ T_WHITE_MUG_ON_PLATE @ T_HAND_ON_WHITE_MUG

    move_to_smooth(env, white_goal, offset=[0, 0, 0.15], steps=100, grip=1.0)
    move_to_smooth(env, white_goal, offset=[0, 0, 0.045], steps=100, grip=1.0)

    gripper_action(env, cmd=-1.0, steps=40)

    move_to_smooth(env, white_goal, offset=[0, 0, 0.20], steps=100, grip=-1.0)

    print("Phase 2: Placing yellow and white mug on right plate...")

    yellow_pick = get_matrix(env, "white_yellow_mug_1_main") @ T_HAND_ON_YELLOW_MUG

    move_to_smooth(env, yellow_pick, offset=[0, 0, 0.15], steps=100, grip=-1.0)
    move_to_smooth(env, yellow_pick, offset=[0, 0, 0.00], steps=100, grip=-1.0)

    gripper_action(env, cmd=1.0, steps=40)

    move_to_smooth(env, yellow_pick, offset=[0, 0, 0.12], steps=100, grip=1.0)

    yellow_goal = get_matrix(env, "plate_2_main") @ T_YELLOW_MUG_ON_PLATE @ T_HAND_ON_YELLOW_MUG

    move_to_smooth(env, yellow_goal, offset=[0, 0, 0.15], steps=100, grip=1.0)
    move_to_smooth(env, yellow_goal, offset=[0, 0, 0.045], steps=100, grip=1.0)

    gripper_action(env, cmd=-1.0, steps=40)

    move_to_smooth(env, yellow_goal, offset=[0, 0, 0.20], steps=100, grip=-1.0)

    print("Mission complete.")

    return True