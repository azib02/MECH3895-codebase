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

# Parent = alphabet_soup_1_main, Child = robot0_right_hand
T_HAND_ON_SOUP = np.array(
    [
        [-0.02517364, 0.99962145, 0.01110136, -0.00104641],
        [0.0048023, 0.01122567, -0.99992546, 0.09557908],
        [-0.99967156, -0.02511845, -0.00508308, -0.00237246],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

# Parent = basket_1_main, Child = alphabet_soup_1_main
T_SOUP_IN_BASKET = np.array(
    [
        [-3.18632422e-02, 2.64218453e-03, 9.99488746e-01, -19.41081034e-04],
        [9.99466934e-01, -7.03147813e-03, 3.18811348e-02, -3.12967981e-02],
        [7.11211910e-03, 9.99971788e-01, -2.41673038e-03, 1.27405247e-01],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


# ---------------------------------------------------------------------------
# Main policy
# ---------------------------------------------------------------------------

def run_solver(env, bddl_file=None):
    """
    Scripted policy for:
    pick_up_the_alphabet_soup_and_place_it_in_the_basket_var_1

    The environment is created by run_collection.py.
    This function only controls the robot.
    """
    print("Phase 1: Placing alphabet soup in basket...")

    soup_grasp = get_matrix(env, "alphabet_soup_1_main") @ T_HAND_ON_SOUP

    move_to_smooth(env, soup_grasp, offset=[0, 0, 0.15], steps=100, grip=-1.0)
    move_to_smooth(env, soup_grasp, offset=[0, 0, 0.00], steps=100, grip=-1.0)

    gripper_action(env, cmd=1.0, steps=40)

    move_to_smooth(env, soup_grasp, offset=[0, 0, 0.20], steps=100, grip=1.0)

    goal_soup = get_matrix(env, "basket_1_main") @ T_SOUP_IN_BASKET @ T_HAND_ON_SOUP

    move_to_smooth(env, goal_soup, offset=[0, 0, 0.15], steps=100, grip=1.0)
    move_to_smooth(env, goal_soup, offset=[0, 0, 0.02], steps=100, grip=1.0)

    gripper_action(env, cmd=-1.0, steps=40)

    move_to_smooth(env, goal_soup, offset=[-0.06, 0, 0.20], steps=100, grip=-1.0)

    print("Mission complete.")

    return True