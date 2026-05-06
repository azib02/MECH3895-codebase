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

# Parent = black_book_1_main, Child = robot0_right_hand
T_HAND_ON_BOOK = np.array(
    [
        [-4.75181300e-02, 9.98851679e-01, -6.11158068e-03, -0.00044307],
        [9.97194511e-01, 4.70832418e-02, -5.81917145e-02, -0.00198328],
        [-5.78371387e-02, -8.85959617e-03, -9.98286719e-01, 0.15302154],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

# Parent = desk_caddy_1_main, Child = black_book_1_main
T_BOOK_IN_CADDY = np.array(
    [
        [0.99956508, 0.02645522, 0.01302983, 0.04273837],
        [-0.02718679, 0.99785207, 0.05959966, 0.01886096],
        [-0.01142512, -0.05992798, 0.99813732, 0.07475523],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


# ---------------------------------------------------------------------------
# Main policy
# ---------------------------------------------------------------------------

def run_solver(env, bddl_file=None):
    """
    Scripted policy for:
    pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy

    The environment is created by run_collection.py.
    This function only controls the robot.
    """
    print("Phase 1: Picking up black book...")

    pick_pose = get_matrix(env, "black_book_1_main") @ T_HAND_ON_BOOK

    move_to_smooth(env, pick_pose, offset=[0, 0, 0.12], steps=100, grip=-1.0)
    move_to_smooth(env, pick_pose, offset=[0, 0, 0.00], steps=100, grip=-1.0)

    gripper_action(env, cmd=1.0, steps=40)

    move_to_smooth(env, pick_pose, offset=[0, 0, 0.20], steps=100, grip=1.0)

    print("Phase 2: Placing book in caddy back compartment...")

    goal_pose = get_matrix(env, "desk_caddy_1_main") @ T_BOOK_IN_CADDY @ T_HAND_ON_BOOK

    move_to_smooth(env, goal_pose, offset=[0, 0, 0.15], steps=100, grip=1.0)
    move_to_smooth(env, goal_pose, offset=[0, 0, 0.01], steps=100, grip=1.0)

    gripper_action(env, cmd=-1.0, steps=40)

    move_to_smooth(env, goal_pose, offset=[0, 0, 0.20], steps=100, grip=-1.0)

    print("Mission complete.")

    return True