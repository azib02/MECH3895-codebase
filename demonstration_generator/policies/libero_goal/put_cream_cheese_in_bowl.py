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

# Parent = cream_cheese_1_main, Child = robot0_right_hand
T_HAND_ON_CHEESE = np.array(
    [
        [-9.98673015e-01, -2.03125733e-03, 5.14595255e-02, -0.03008898],
        [-2.03371463e-03, 9.99997932e-01, 4.60956228e-06, 0.00012527],
        [-5.14594285e-02, -1.00050544e-04, -9.98675081e-01, 0.03867907],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

# Parent = akita_black_bowl_1_main, Child = cream_cheese_1_main
T_CHEESE_IN_BOWL = np.array(
    [
        [5.82986968e-04, -9.99908385e-01, -1.35233666e-02, 0.00173612],
        [9.99764155e-01, 8.76379403e-04, -2.16994518e-02, 0.01990440],
        [2.17093154e-02, -1.35075266e-02, 9.99673073e-01, 0.07887313],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


# ---------------------------------------------------------------------------
# Main policy
# ---------------------------------------------------------------------------

def run_solver(env, bddl_file=None):
    """
    Scripted policy for:
    put_the_cream_cheese_in_the_bowl

    The environment is created by run_collection.py.
    This function only controls the robot.
    """
    print("Phase 1: Picking up cream cheese...")

    pick_pose = get_matrix(env, "cream_cheese_1_main") @ T_HAND_ON_CHEESE

    move_to_smooth(env, pick_pose, offset=[0, 0, 0.10], steps=100, grip=-1.0)
    move_to_smooth(env, pick_pose, offset=[0, 0, 0.00], steps=100, grip=-1.0)

    gripper_action(env, cmd=1.0, steps=40)

    move_to_smooth(env, pick_pose, offset=[0, 0, 0.15], steps=100, grip=1.0)

    print("Phase 2: Placing cream cheese in bowl...")

    goal_pose = get_matrix(env, "akita_black_bowl_1_main") @ T_CHEESE_IN_BOWL @ T_HAND_ON_CHEESE

    move_to_smooth(env, goal_pose, offset=[0, 0, 0.10], steps=100, grip=1.0)
    move_to_smooth(env, goal_pose, offset=[0, 0, 0.01], steps=100, grip=1.0)

    gripper_action(env, cmd=-1.0, steps=40)

    move_to_smooth(env, goal_pose, offset=[0, 0, 0.20], steps=100, grip=-1.0)

    print("Mission complete.")

    return True