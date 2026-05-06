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

# Parent = wine_bottle_1_main, Child = robot0_right_hand
T_HAND_ON_BOTTLE = np.array(
    [
        [-0.996276425, -0.03236949, 0.079909336, -0.00705844],
        [-0.032258746, 0.99947596, 0.002676779, -0.00043849],
        [-0.079954107, 0.00008903, -0.996798542, 0.12583424],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

# Parent = wooden_cabinet_1_main, Child = wine_bottle_1_main
T_BOTTLE_ON_CABINET = np.array(
    [
        [0.99633326, -0.08537235, 0.00562091, -0.00394678],
        [0.08540065, 0.99633412, -0.00500491, -0.01343106],
        [-0.00517303, 0.00546659, 0.99997168, 0.22244491],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


# ---------------------------------------------------------------------------
# Main policy
# ---------------------------------------------------------------------------

def run_solver(env, bddl_file=None):
    """
    Scripted policy for:
    put_the_wine_bottle_on_top_of_the_cabinet

    The environment is created by run_collection.py.
    This function only controls the robot.
    """
    print("Phase 1: Picking up wine bottle...")

    pick_pose = get_matrix(env, "wine_bottle_1_main") @ T_HAND_ON_BOTTLE

    move_to_smooth(env, pick_pose, offset=[0, 0, 0.15], steps=100, grip=-1.0)
    move_to_smooth(env, pick_pose, offset=[0, 0, 0.00], steps=100, grip=-1.0)

    gripper_action(env, cmd=1.0, steps=40)

    move_to_smooth(env, pick_pose, offset=[0, 0, 0.30], steps=100, grip=1.0)

    print("Phase 2: Placing wine bottle on cabinet...")

    goal_pose = get_matrix(env, "wooden_cabinet_1_main") @ T_BOTTLE_ON_CABINET @ T_HAND_ON_BOTTLE

    move_to_smooth(env, goal_pose, offset=[0, 0, 0.15], steps=100, grip=1.0)
    move_to_smooth(env, goal_pose, offset=[0, 0, 0.01], steps=100, grip=1.0)

    gripper_action(env, cmd=-1.0, steps=40)

    move_to_smooth(env, goal_pose, offset=[0, 0, 0.20], steps=100, grip=-1.0)

    print("Mission complete.")

    return True