# MECH3895 Codebase

This repository contains the codebase for the MECH3895 individual engineering project:

**Beyond LIBERO: LLM-Augmented Data for Vision-Language-Action Models**

The project explores how large language models can be used to augment LIBERO robotic manipulation tasks and support demonstration generation for VLA model training.

---

## Repository Structure

```text
MECH3895-codebase/
├── bddl_generator/
├── demonstration_generator/
└── script_generator/
```

---

## 1. BDDL Generator

Folder:

```text
bddl_generator/
```

This module generates augmented LIBERO BDDL task files.

It applies transformations such as:

```text
- task language rephrasing
- object region shifting
- yaw randomisation
- standalone object location swapping
- validation of generated BDDL files
```

See:

```text
bddl_generator/README.md
```

---

## 2. Demonstration Generator

Folder:

```text
demonstration_generator/
```

This module runs scripted LIBERO policies and collects robot demonstration data.

It includes:

```text
- scripted policy modules
- HDF5 demonstration collection
- dataset restructuring
- raw and processed demo checking tools
- video and frame extraction tools
- bad-demo deletion tools
```

See:

```text
demonstration_generator/README.md
```

---

## 3. Script Generator

Folder:

```text
script_generator/
```

This module supports AI-assisted policy script generation.

It stores successful scripts and matrices in a PostgreSQL database, retrieves similar previous tasks using embeddings, and uses an LLM to generate new policy drafts from BDDL files.

It includes:

```text
- policy archiving
- policy retrieval
- BDDL-based policy generation
- matrix JSON storage
- PostgreSQL + pgvector schema
```

See:

```text
script_generator/README.md
```

---

## Overall Workflow

```text
1. Use bddl_generator to create augmented LIBERO BDDL tasks.
2. Use script_generator to help create or retrieve matching scripted policies.
3. Use demonstration_generator to run those policies and collect demonstrations.
4. Restructure successful demonstrations for downstream VLA training.
```

---

## Setup Notes

Each folder has its own:

```text
README.md
requirements.txt
.gitignore
```

Install requirements separately inside each module as needed.

Example:

```bash
cd bddl_generator
pip install -r requirements.txt
```

```bash
cd ../demonstration_generator
pip install -r requirements.txt
```

```bash
cd ../script_generator
pip install -r requirements.txt
```

LIBERO, robosuite, MuJoCo, and related robotics dependencies should be installed separately according to their official installation instructions.

---

## Security Notes

Real environment files are not included.

Do not commit:

```text
.env
API keys
database passwords
HDF5 datasets
videos
generated temporary files
```

Only `.env.example` files are provided as templates.

---

## Project Context

This repository was developed as part of a final-year Mechatronics and Robotics engineering project at the University of Leeds.

The code supports experiments on LLM-augmented task generation and scripted demonstration collection for robotic manipulation tasks in LIBERO.