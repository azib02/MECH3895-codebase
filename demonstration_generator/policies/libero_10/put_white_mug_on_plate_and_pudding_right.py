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

# Parent = chocolate_pudding_1_main, Child = robot0_right_hand
T_HAND_ON_PUDDING = np.array(
    [
        [-0.99816025, 0.02084189, 0.05693618, -0.00374868],
        [0.01977707, 0.99962, -0.01920193, 0.00177477],
        [-0.05731475, -0.01804057, -0.99819315, 0.09413056],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

# Parent = plate_1_main, Child = chocolate_pudding_1_main
T_PUDDING_ON_PLATE = np.array(
    [
        [3.48969162e-02, -9.99313726e-01, -1.24210216e-02, 0.11678153],
        [9.99390624e-01, 3.49037347e-02, -3.32527429e-04, 0.01680664],
        [7.65839267e-04, -1.24018484e-02, 9.99922801e-01, 0.01124958],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

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


# ---------------------------------------------------------------------------
# Main policy
# ---------------------------------------------------------------------------

def run_solver(env, bddl_file=None):
    """
    Scripted policy for:
    put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate

    The environment is created by run_collection.py.
    This function only controls the robot.
    """
    print("Phase 1: Placing white mug on plate...")

    mug_pick = get_matrix(env, "porcelain_mug_1_main") @ T_HAND_ON_WHITE_MUG

    move_to_smooth(env, mug_pick, offset=[0, 0, 0.15], steps=100, grip=-1.0)
    move_to_smooth(env, mug_pick, offset=[0, 0, 0.00], steps=100, grip=-1.0)

    gripper_action(env, cmd=1.0, steps=40)

    move_to_smooth(env, mug_pick, offset=[0, 0, 0.20], steps=100, grip=1.0)

    mug_goal = get_matrix(env, "plate_1_main") @ T_WHITE_MUG_ON_PLATE @ T_HAND_ON_WHITE_MUG

    move_to_smooth(env, mug_goal, offset=[0, 0, 0.15], steps=100, grip=1.0)
    move_to_smooth(env, mug_goal, offset=[0, 0, 0.05], steps=100, grip=1.0)

    gripper_action(env, cmd=-1.0, steps=40)

    move_to_smooth(env, mug_goal, offset=[0, 0, 0.20], steps=100, grip=-1.0)

    print("Phase 2: Placing chocolate pudding to the right of plate...")

    pudding_pick = get_matrix(env, "chocolate_pudding_1_main") @ T_HAND_ON_PUDDING

    move_to_smooth(env, pudding_pick, offset=[0, 0, 0.15], steps=100, grip=-1.0)
    move_to_smooth(env, pudding_pick, offset=[0, 0, 0.00], steps=100, grip=-1.0)

    gripper_action(env, cmd=1.0, steps=40)

    pudding_goal = get_matrix(env, "plate_1_main") @ T_PUDDING_ON_PLATE @ T_HAND_ON_PUDDING

    move_to_smooth(env, pudding_goal, offset=[0, 0.05, 0.15], steps=100, grip=1.0)
    move_to_smooth(env, pudding_goal, offset=[0, 0.00, 0.01], steps=100, grip=1.0)

    gripper_action(env, cmd=-1.0, steps=40)

    move_to_smooth(env, pudding_goal, offset=[0, 0.15, 0.20], steps=100, grip=-1.0)

    print("Mission complete.")

    return True