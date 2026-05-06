# Script Generator

This folder contains the AI-assisted policy script generator used in the project:

**Beyond LIBERO: LLM-Augmented Data for VLAs**

The script generator stores successful LIBERO scripted policies in a PostgreSQL database, retrieves similar past examples using embeddings, and uses an LLM to generate new policy scripts for new BDDL tasks.

The purpose of this tool is to speed up the creation of scripted policies by reusing previous successful scripts, matrices, object transforms, and task descriptions.

---

## Folder Structure

```text
script_generator/
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
│
├── database/
│   └── schema.sql
│
├── generator/
│   ├── __init__.py
│   ├── config.py
│   ├── db.py
│   ├── embeddings.py
│   └── prompts.py
│
├── tools/
│   ├── archive_policy.py
│   ├── retrieve_policy.py
│   └── generate_policy.py
│
├── input_bddl/
├── reference_scripts/
└── generated_scripts/
```

---

## Purpose

This tool acts as a small memory system for policy generation.

It can:

```text
- archive successful policy scripts
- store reusable transformation matrices
- embed task descriptions for semantic search
- retrieve old policies by task name
- find similar previous tasks
- generate new policy scripts from new BDDL files
```

---

## Setup

Install the requirements:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
copy .env.example .env
```

Then fill in:

```env
GROQ_API_KEY=your_groq_key_here

DB_NAME=BDDL-Script
DB_USER=postgres
DB_PASSWORD=your_database_password_here
DB_HOST=127.0.0.1
DB_PORT=5433

EMBEDDING_MODEL=all-mpnet-base-v2
GROQ_MODEL_SUMMARY=llama-3.1-8b-instant
GROQ_MODEL_GENERATE=llama-3.3-70b-versatile
```

Do not commit the real `.env` file.

---

## Database Setup

This tool uses PostgreSQL with the `pgvector` extension.

Run the schema file once:

```bash
psql -U postgres -h 127.0.0.1 -p 5433 -d BDDL-Script -f database/schema.sql
```

On Windows PowerShell:

```powershell
psql -U postgres -h 127.0.0.1 -p 5433 -d BDDL-Script -f database/schema.sql
```

The schema creates three tables:

```text
tasks
matrices
scripts
```

---

## Main Tools

### 1. Archive a Policy

Use this after a scripted policy has been tested successfully.

```bash
python tools/archive_policy.py ^
  --task-name "pick_black_bowl_place_on_plate" ^
  --description "Pick up the black bowl and place it on the plate." ^
  --script reference_scripts/pick_black_bowl_from_table_center_place_on_plate.py ^
  --matrices reference_scripts/pick_black_bowl_from_table_center_place_on_plate_matrices.json
```

This saves:

```text
- task name
- cleaned task description
- task embedding
- policy script code
- reusable matrices
```

---

### 2. Retrieve a Policy

Use this to recover a stored policy from the database.

```bash
python tools/retrieve_policy.py ^
  --task-name "pick_black_bowl_place_on_plate" ^
  --output generated_scripts/recovered_pick_black_bowl.py
```

---

### 3. Generate a New Policy

Use this to generate a new policy from a new BDDL file.

```bash
python tools/generate_policy.py ^
  --bddl-file input_bddl/new_task.bddl ^
  --output generated_scripts/new_policy.py ^
  --reference-script reference_scripts/pick_black_bowl_from_table_center_place_on_plate.py ^
  --notes "Use the same move_to_smooth style as the cleaned policies."
```

The generator:

```text
1. reads the new BDDL file
2. embeds the BDDL content
3. searches the database for the most similar previous task
4. retrieves stored scripts and matrices
5. builds an LLM prompt
6. saves the generated policy script
```

---

## Matrix JSON Files

Matrices are stored using JSON files.

Example:

```json
[
  {
    "label": "T_HAND_ON_BOWL",
    "data": [
      [0.996954, -0.02150385, -0.07496874, 0.0087228],
      [-0.02373346, -0.99929827, -0.02897746, 0.05306893],
      [-0.074293, 0.03066846, -0.99676476, 0.05358268],
      [0.0, 0.0, 0.0, 1.0]
    ]
  },
  {
    "label": "T_BOWL_ON_PLATE",
    "data": [
      [0.99234336, -0.12171498, 0.02097882, 0.00275265],
      [0.12132078, 0.99242876, 0.01914206, -0.0119236],
      [-0.02314986, -0.01645033, 0.99959665, 0.07515369],
      [0.0, 0.0, 0.0, 1.0]
    ]
  }
]
```

Recommended naming:

```text
reference_scripts/
├── pick_black_bowl_from_table_center_place_on_plate.py
└── pick_black_bowl_from_table_center_place_on_plate_matrices.json
```

---

## Generated Scripts

Generated scripts are saved into:

```text
generated_scripts/
```

After checking the generated policy, copy it into the demonstration generator:

```text
demonstration_generator/policies/
```

Example:

```text
demonstration_generator/policies/libero_spatial/new_policy.py
```

Then test it using:

```bash
python tools/test_and_record_policy.py ^
  --bddl-file input_bddl/new_task.bddl ^
  --policy libero_spatial.new_policy ^
  --output videos/new_policy_test.mp4
```

This test command should be run from inside the `demonstration_generator` folder.

---

## Relationship to the Other Folders

The full project workflow is:

```text
1. bddl_generator
   Creates augmented BDDL task files.

2. script_generator
   Helps generate new policy scripts for those BDDL files.

3. demonstration_generator
   Runs policies in LIBERO and collects demonstration data.
```

The script generator does not run robot environments itself. It only creates or retrieves policy scripts.

---

## Requirements

The main packages are:

```text
groq
python-dotenv
psycopg2-binary
sentence-transformers
numpy
pgvector
```

These are listed in:

```text
requirements.txt
```

---

## Security Notes

Do not commit:

```text
.env
database passwords
API keys
generated scripts if they are experimental
large datasets
```

Only commit:

```text
.env.example
source code
README.md
schema.sql
small reference examples if needed
```

If a real API key or database password was committed accidentally, rotate it immediately.

---

## Notes

This tool is experimental and should be used as an assistant, not as a fully automatic replacement for testing.

Every generated policy should still be tested visually using the demonstration generator before collecting final demonstrations.