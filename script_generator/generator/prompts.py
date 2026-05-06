def build_generation_prompt(
    bddl_content,
    manual_matrices,
    manual_script_content,
    additional_notes,
    similar_task_name,
    similar_script,
    database_matrices,
):
    return f"""
You are a precise robotics policy generator for a Panda robot in LIBERO.

Your job is to generate ONE Python policy module that contains a run_solver(env, bddl_file=None) function.

The generated script must match the same policy style used in this project:
- import numpy as np
- import only robosuite.utils.transform_utils helpers needed for movement
- do NOT create the environment
- do NOT import TASK_MAPPING
- do NOT import load_controller_config
- do NOT call env.reset()
- do NOT call env.close()
- use get_matrix(env, body_name)
- use move_to_smooth(...)
- use gripper_action(...)
- use getattr(env, "has_renderer", False) before rendering
- return True at the end of run_solver

--- MISSION GOAL FROM BDDL ---
{bddl_content}

--- USER PROVIDED DATA, HIGHEST PRIORITY ---
Manual matrices:
{manual_matrices if manual_matrices else "None provided"}

Manual reference script:
{manual_script_content if manual_script_content else "None provided"}

Extra instructions:
{additional_notes if additional_notes else "None"}

--- DATABASE MEMORY ---
Most similar previous task:
{similar_task_name if similar_task_name else "N/A"}

Stored code template:
{similar_script if similar_script else "N/A"}

Stored matrices:
{database_matrices if database_matrices else "None available"}

--- CRITICAL RULES ---
1. Only generate code for the task described in the BDDL.
2. Use the similar script only as structure. Remove irrelevant old actions.
3. Prefer manual matrices over database matrices.
4. Preserve matrix numeric values exactly when using them.
5. Use object/body names that appear in the BDDL or reference script.
6. Output ONLY Python code. No markdown, no backticks, no explanation.
"""