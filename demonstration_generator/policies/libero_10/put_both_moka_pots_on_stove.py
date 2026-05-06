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

# Parent = moka_pot_*_main, Child = robot0_right_hand
T_HAND_ON_MOKA = np.array(
    [
        [0.0468999, -0.99776395, -0.04761826, 0.00474117],
        [-0.99887045, -0.04720934, 0.00539408, 0.07027523],
        [-0.00763004, 0.04731149, -0.99885104, 0.12649014],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

# Parent = flat_stove_1_burner, Child = moka_pot_*_main
T_MOKA_ON_STOVE = np.array(
    [
        [-9.96476920e-01, 5.99797513e-02, 5.86189138e-02, -6.20754334e-04],
        [-5.84217278e-02, -9.97900864e-01, 2.79422147e-02, 2.69366813e-03],
        [6.01718318e-02, 2.44191538e-02, 9.97889300e-01, 0.09376556],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


# ---------------------------------------------------------------------------
# Main policy
# ---------------------------------------------------------------------------

def run_solver(env, bddl_file=None):
    """
    Scripted policy for:
    put_both_moka_pots_on_the_stove

    The environment is created by run_collection.py.
    This function only controls the robot.
    """
    print("Phase 1: Moka pot 1 to left side of burner...")

    m1_grasp = get_matrix(env, "moka_pot_1_main") @ T_HAND_ON_MOKA

    move_to_smooth(env, m1_grasp, offset=[0, 0, 0.15], steps=100, grip=-1.0)
    move_to_smooth(env, m1_grasp, offset=[0, 0, 0.00], steps=100, grip=-1.0)

    gripper_action(env, cmd=1.0, steps=40)

    m1_goal = get_matrix(env, "flat_stove_1_burner") @ T_MOKA_ON_STOVE
    m1_hand_goal = m1_goal @ T_HAND_ON_MOKA

    move_to_smooth(env, m1_hand_goal, offset=[-0.05, 0, 0.15], steps=100, grip=1.0)
    move_to_smooth(env, m1_hand_goal, offset=[-0.05, 0, -0.04], steps=100, grip=1.0)

    gripper_action(env, cmd=-1.0, steps=40)

    move_to_smooth(env, m1_hand_goal, offset=[-0.05, 0, 0.10], steps=100, grip=-1.0)

    print("Phase 2: Moka pot 2 to right side of burner...")

    m2_grasp = get_matrix(env, "moka_pot_2_main") @ T_HAND_ON_MOKA

    move_to_smooth(env, m2_grasp, offset=[0, 0, 0.15], steps=100, grip=-1.0)
    move_to_smooth(env, m2_grasp, offset=[0, 0, 0.00], steps=100, grip=-1.0)

    gripper_action(env, cmd=1.0, steps=40)

    m2_goal = get_matrix(env, "flat_stove_1_burner") @ T_MOKA_ON_STOVE
    m2_hand_goal = m2_goal @ T_HAND_ON_MOKA

    move_to_smooth(env, m2_hand_goal, offset=[0.05, 0, 0.15], steps=100, grip=1.0)
    move_to_smooth(env, m2_hand_goal, offset=[0.05, 0, 0.00], steps=100, grip=1.0)

    gripper_action(env, cmd=-1.0, steps=40)

    move_to_smooth(env, m2_hand_goal, offset=[0.05, 0, 0.20], steps=100, grip=-1.0)

    print("Mission complete.")

    return True