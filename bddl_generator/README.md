# BDDL Generator

This folder contains the BDDL augmentation pipeline used in the project:

**Beyond LIBERO: LLM-Augmented Data for VLAs**

The generator takes existing LIBERO BDDL task files and creates augmented task variants by changing the language instruction, shifting movable object regions, randomising yaw angles, and swapping standalone object locations where possible.

The goal is to create more diverse LIBERO-compatible task definitions while keeping the generated files valid and usable for demonstration collection.

---

## Folder Structure

```text
bddl_generator/
├── run_pipeline.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
│
├── modules/
│   ├── __init__.py
│   ├── parser.py
│   ├── validator.py
│   ├── proximity_validator.py
│   ├── grouping.py
│   ├── rephrase.py
│   ├── shift.py
│   ├── yaw.py
│   └── swap.py
│
├── input_bddl/
├── output_bddl/
└── attempts/
```

---

## Main Pipeline

The main script is:

```bash
python run_pipeline.py
```

The pipeline applies these stages:

```text
1. Parse the input BDDL file
2. Rephrase the task language
3. Shift movable object regions
4. Randomise yaw values
5. Swap standalone object locations where possible
6. Validate the generated BDDL
7. Save valid outputs
```

---

## Setup

Install the required packages:

```bash
pip install -r requirements.txt
```

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Then add your Groq API key:

```env
GROQ_API_KEY=your_actual_key_here
```

On Windows PowerShell, you can create it manually:

```powershell
copy .env.example .env
```

---

## Input Files

Place original LIBERO BDDL files inside:

```text
input_bddl/
```

Example:

```text
input_bddl/
└── KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.bddl
```

---

## Output Files

Generated BDDL files are saved into:

```text
output_bddl/
```

Intermediate attempts are saved into:

```text
attempts/
```

The `attempts/` folder is useful for debugging failed generations or checking how each augmentation stage changed the task.

---

## Modules

### `parser.py`

Parses BDDL files and extracts important sections such as:

```text
objects
fixtures
regions
initial state
goal state
language instruction
```

### `validator.py`

Checks whether generated BDDL files are structurally valid.

The validator checks:

```text
- required objects and fixtures still exist
- object names have not been broken
- existing region ranges contain valid coordinate values
- coordinate expressions are complete
- coordinate ordering is valid
- parentheses are balanced
```

### `proximity_validator.py`

Checks object and region proximity constraints to reduce invalid scene layouts.

### `grouping.py`

Groups related task information so the LLM has cleaner structured context.

### `rephrase.py`

Uses an LLM to generate alternative natural language task instructions.

### `shift.py`

Applies coordinate shifts to movable object regions.

### `yaw.py`

Randomises yaw angles for objects or regions where this is safe.

### `swap.py`

Swaps standalone object locations when the task structure allows it.

---

## Running the Generator

From inside the `bddl_generator` folder:

```bash
python run_pipeline.py
```

On Windows PowerShell:

```powershell
python run_pipeline.py
```

The script reads from:

```text
input_bddl/
```

and writes valid generated files to:

```text
output_bddl/
```

---

## Requirements

The main external packages are:

```text
groq
python-dotenv
```

These are listed in:

```text
requirements.txt
```

---

## Notes

This generator does not collect robot demonstrations. It only creates augmented BDDL task files.

To collect demonstrations from generated BDDL files, use the separate:

```text
demonstration_generator/
```

The intended workflow is:

```text
1. Generate augmented BDDL files with bddl_generator
2. Copy valid BDDL files into demonstration_generator/input_bddl
3. Run scripted policies with demonstration_generator
4. Restructure successful demonstrations for training
```

---

## GitHub Notes

The following folders are intentionally kept but their generated contents are ignored by Git:

```text
input_bddl/
output_bddl/
attempts/
```

Each folder should contain a `.gitkeep` file so that the empty folder appears on GitHub.

Do not commit:

```text
.env
generated BDDL files
temporary attempts
```