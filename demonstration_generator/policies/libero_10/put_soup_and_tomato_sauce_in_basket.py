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

# Parent = alphabet_soup_1_main, Child = robot0_right_hand
T_HAND_ON_SOUP = np.array(
    [
        [-0.02517364, 0.99962145, 0.01110136, -0.00104641],
        [0.0048023, 0.01122567, -0.99992546, 0.13557908],
        [-0.99967156, -0.02511845, -0.00508308, -0.00237246],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

# Parent = basket_1_main, Child = alphabet_soup_1_main
T_SOUP_IN_BASKET = np.array(
    [
        [-3.18632422e-02, 2.64218453e-03, 9.99488746e-01, -0.00194108],
        [9.99466934e-01, -7.03147813e-03, 3.18811348e-02, -0.03129680],
        [7.11211910e-03, 9.99971788e-01, -2.41673038e-03, 0.12740525],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

# Parent = tomato_sauce_1_main, Child = robot0_right_hand
T_HAND_ON_SAUCE = np.array(
    [
        [-0.99881681, -0.01378172, 0.04663740, 0.00103251],
        [-0.04678371, 0.01049134, -0.99884995, 0.10350568],
        [0.01327658, -0.99984999, -0.01112368, -0.00076047],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

# Parent = basket_1_main, Child = tomato_sauce_1_main
T_SAUCE_IN_BASKET = np.array(
    [
        [0.00882323, 0.00077250, 0.99996078, 0.01342727],
        [0.99996031, 0.00122815, -0.00882417, 0.02169642],
        [-0.00123492, 0.99999895, -0.00076163, 0.11126627],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


# ---------------------------------------------------------------------------
# Main policy
# ---------------------------------------------------------------------------

def run_solver(env, bddl_file=None):
    """
    Scripted policy for:
    put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket

    The environment is created by run_collection.py.
    This function only controls the robot.
    """
    print("Phase 1: Picking up alphabet soup...")

    soup_pick = get_matrix(env, "alphabet_soup_1_main") @ T_HAND_ON_SOUP

    move_to_smooth(env, soup_pick, offset=[0, 0, 0.15], steps=100, grip=-1.0)
    move_to_smooth(env, soup_pick, offset=[0, 0, 0.00], steps=100, grip=-1.0)

    gripper_action(env, cmd=1.0, steps=40)

    move_to_smooth(env, soup_pick, offset=[0, 0, 0.20], steps=100, grip=1.0)

    print("Phase 2: Placing soup in basket...")

    soup_goal = get_matrix(env, "basket_1_main") @ T_SOUP_IN_BASKET @ T_HAND_ON_SOUP

    move_to_smooth(env, soup_goal, offset=[0, 0, 0.15], steps=100, grip=1.0)
    move_to_smooth(env, soup_goal, offset=[0, 0, 0.02], steps=100, grip=1.0)

    gripper_action(env, cmd=-1.0, steps=40)

    move_to_smooth(env, soup_goal, offset=[0, -0.06, 0.20], steps=100, grip=-1.0)

    print("Phase 3: Picking up tomato sauce...")

    sauce_pick = get_matrix(env, "tomato_sauce_1_main") @ T_HAND_ON_SAUCE

    move_to_smooth(env, sauce_pick, offset=[0, 0, 0.15], steps=100, grip=-1.0)
    move_to_smooth(env, sauce_pick, offset=[0, 0, 0.00], steps=100, grip=-1.0)

    gripper_action(env, cmd=1.0, steps=40)

    move_to_smooth(env, sauce_pick, offset=[0, 0, 0.20], steps=100, grip=1.0)

    print("Phase 4: Placing tomato sauce in basket...")

    sauce_goal = get_matrix(env, "basket_1_main") @ T_SAUCE_IN_BASKET @ T_HAND_ON_SAUCE

    move_to_smooth(env, sauce_goal, offset=[0, 0, 0.15], steps=100, grip=1.0)
    move_to_smooth(env, sauce_goal, offset=[0, 0, 0.02], steps=100, grip=1.0)

    gripper_action(env, cmd=-1.0, steps=40)

    move_to_smooth(env, sauce_goal, offset=[0, 0.06, 0.20], steps=100, grip=-1.0)

    print("Mission complete.")

    return True