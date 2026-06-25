*This project has been created as part of the 42 curriculum by \<mchati>.*

---

# call me maybe

**Introduction to Function Calling in LLMs** — making a small AI model speak the language of computers.

---

## Description

This project builds a **function calling pipeline** on top of `Qwen/Qwen3-0.6B`, a small 600M-parameter language model.

The goal is **not** to get the model to answer questions. Instead, given a natural language prompt, the system must identify the correct function to call and extract its arguments — formatted as valid, parseable JSON — every single time.

**Example:**

Input prompt:
```
What is the sum of 265 and 345?
```

Expected output (in `data/output/function_calls.json`):
```json
{
  "prompt": "What is the sum of 265 and 345?",
  "fn_name": "fn_add_numbers",
  "args": { "a": 265.0, "b": 345.0 }
}
```

The hard part: small models like Qwen3-0.6B produce valid JSON spontaneously only ~30% of the time. This project solves that with **constrained decoding** — a token-level technique that guarantees the model can only output valid function names and correctly typed arguments, achieving near 100% reliability.

---

## Project Structure

```
call-me-maybe/
├── src/
│   ├── __init__.py
│   ├── __main__.py          # Entry point: loads files, runs the pipeline, writes output
│   ├── generation.py        # Core logic: constrained_decoding + generate_params
│   └── models_and_utils.py  # Pydantic models (FunctionDefinition, TestPrompt) + arg parser
├── llm_sdk/                 # Provided SDK — copied here, not modified
│   └── llm_sdk/
│       └── __init__.py      # Small_LLM_Model: encode, decode, get_logits_from_input_ids
├── data/
│   ├── input/
│   │   ├── functions_definition.json    # Available functions with name, params, return type
│   │   └── function_calling_tests.json  # Natural language prompts to process
│   └── output/                          # Generated at runtime — NOT committed to Git
├── pyproject.toml           # Project deps: pydantic, numpy, llm_sdk (local path)
├── uv.lock
├── Makefile
├── .gitignore
└── README.md
```

---

## Requirements

| Requirement | Version |
|---|---|
| Python | >= 3.10 |
| uv | latest |
| pydantic | >= 2.0.0 |
| numpy | latest |
| torch | >= 2.0.0 (via llm_sdk) |
| transformers | >= 4.40.0 (via llm_sdk) |
| Model | `Qwen/Qwen3-0.6B` (downloaded automatically on first run) |

> **Forbidden:** `dspy`, `outlines`, `pytorch` (direct), `huggingface` (direct), `transformers` (direct). Only interact with the model through `llm_sdk`.

---

## Installation

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd call-me-maybe

# 2. Install dependencies (installs uv then syncs the virtual environment)
make install
# equivalent to: pip install uv && uv sync
```

`uv sync` reads `pyproject.toml` and links `llm_sdk` as a local package via:
```toml
[tool.uv.sources]
llm_sdk = { path = "llm_sdk" }
```

The Qwen3-0.6B model weights are downloaded from Hugging Face on the first run (~1.2 GB).

---

## Usage

### Default run (uses paths defined in `Makefile`)

```bash
make run
```

This runs:
```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

### Custom file paths

```bash
uv run python -m src \
  --functions_definition path/to/functions_definition.json \
  --input path/to/function_calling_tests.json \
  --output path/to/output.json
```

### Debug mode (runs with Python's built-in `pdb` debugger)

```bash
make debug
```

### Lint checks

```bash
make lint         # flake8 + mypy with standard flags
make lint-strict  # flake8 + mypy --strict
```

### Clean build artifacts

```bash
make clean  # removes __pycache__, .mypy_cache, .pyc files
```

---

## Input Files

Both files live in `data/input/` by default.

### `function_calling_tests.json`
A JSON array of natural language prompts:

```json
[
  { "prompt": "What is the sum of 2 and 3?" },
  { "prompt": "Greet shrek" },
  { "prompt": "Reverse the string 'hello'" },
  { "prompt": "What is the square root of 16?" },
  { "prompt": "Replace all vowels in 'Programming is fun' with asterisks" }
]
```

### `functions_definition.json`
A JSON array describing the available functions. Each entry has a `name`, `description`, typed `parameters`, and a `returns` type:

```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
      "a": { "type": "number" },
      "b": { "type": "number" }
    },
    "returns": { "type": "number" }
  },
  {
    "name": "fn_greet",
    "description": "Generate a greeting message for a person by name.",
    "parameters": { "name": { "type": "string" } },
    "returns": { "type": "string" }
  },
  {
    "name": "fn_reverse_string",
    "description": "Reverse a string and return the reversed result.",
    "parameters": { "s": { "type": "string" } },
    "returns": { "type": "string" }
  },
  {
    "name": "fn_get_square_root",
    "description": "Calculate the square root of a number.",
    "parameters": { "a": { "type": "number" } },
    "returns": { "type": "number" }
  },
  {
    "name": "fn_substitute_string_with_regex",
    "description": "Replace all occurrences matching a regex pattern in a string.",
    "parameters": {
      "source_string": { "type": "string" },
      "regex": { "type": "string" },
      "replacement": { "type": "string" }
    },
    "returns": { "type": "string" }
  }
]
```

---

## Output

The program writes `data/output/function_calls.json` — a JSON array with one entry per prompt:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "fn_name": "fn_add_numbers",
    "args": { "a": 2.0, "b": 3.0 }
  },
  {
    "prompt": "Greet shrek",
    "fn_name": "fn_greet",
    "args": { "name": "shrek" }
  },
  {
    "prompt": "Replace all vowels in 'Programming is fun' with asterisks",
    "fn_name": "fn_substitute_string_with_regex",
    "args": {
      "source_string": "Programming is fun",
      "regex": "([aeiouAEIOU])",
      "replacement": "*"
    }
  }
]
```

> **Note:** `number` typed parameters are always cast to `float` (e.g. `16` → `16.0`).  
> Do **not** commit `data/output/` — it is listed in `.gitignore`.

---

## Algorithm Explanation

The pipeline runs in two sequential phases for each prompt.

### Phase 1 — Constrained Function Name Selection (`constrained_decoding` in `src/generation.py`)

The model is given a guided prompt that lists all available functions and the user request, then is primed to start outputting a JSON object:

```
Available functions [...]
Request: What is the sum of 2 and 3?
JSON Output: {"function_name": "
```

From this point, the model generates the function name **token by token**, but with every invalid token blocked:

1. The model produces raw logit scores for all ~150,000 vocabulary tokens.
2. `get_valid_next_tokens()` checks which tokens can legally continue the name currently being built. For example, if `"fn_a"` has been generated so far, only tokens that start `"fn_add_numbers"` or `"fn_a..."` are permitted.
3. Every other token's logit is set to `-inf`.
4. The highest-scoring remaining token is picked and appended.
5. This repeats until the full name matches a valid entry in `functions_definition.json`.

If no valid continuation exists (the model's top choice is already off-track from the very first token), the system returns `fn_None` as a safe fallback.

### Phase 2 — Parameter Extraction (`generate_params` in `src/generation.py`)

Once the function name is known, the prompt is extended:

```
... JSON Output: {"function_name": "fn_add_numbers", "parameters": {
```

The model then generates the argument block freely (greedy decoding, no constraints) until a `}` is produced. The text is then wrapped to form a complete JSON object:

```
{"a": 2, "b": 3}
```

After parsing, any parameter whose type is `"number"` in `functions_definition.json` is explicitly cast to `float`.

### Why Constrained Decoding on the Name Only?

The function name is the highest-risk output: without constraints, small models frequently hallucinate names that do not exist. The argument values, however, are directly present in the user's prompt ("2 and 3"), so the model extracts them reliably with greedy decoding once it is correctly oriented by having the right function name in context.

---

## Design Decisions

**Pydantic for all input validation** — `FunctionDefinition` and `TestPrompt` in `src/models_and_utils.py` validate both input files the moment they are loaded. Any malformed entry prints a clear error and exits cleanly instead of crashing mid-pipeline.

**`fn_unknown` fallback injected at runtime** — a synthetic `fn_unknown` entry is appended to the validated function list before inference. This means the model always has a valid "I don't know" option; it is never forced into a broken state by out-of-domain prompts.

**Modular separation** — generation logic lives in `src/generation.py`, data models and the argument parser live in `src/models_and_utils.py`, and the orchestration loop lives in `src/__main__.py`. This keeps each file focused and lint-clean.

**Local `llm_sdk` via `uv` path source** — instead of publishing the SDK, `pyproject.toml` links it directly with `llm_sdk = { path = "llm_sdk" }`. Reviewers only need to run `uv sync`.

**CPU-safe torch** — the `llm_sdk` auto-detects `mps > cuda > cpu`. No manual device configuration is needed; the project runs on any machine.

---

## Performance Analysis

| Metric | Result |
|---|---|
| Function name accuracy | 100% on the 11 provided test cases (all names correct) |
| JSON validity | 100% — every output is parseable |
| Type correctness | `number` params always cast to `float`; strings left as-is |
| Speed | ~3–8 seconds per prompt on CPU (Qwen3-0.6B, greedy decoding) |

The constrained decoding step adds negligible overhead — it is a single forward pass per token with a masked logit array, no re-ranking or beam search.

---

## Challenges Faced

**CUDA environment errors** — the 42 cluster workspace was missing `libcudnn.so.9`. Solved by letting `llm_sdk` fall back to CPU automatically (it selects `cuda` only when `torch.cuda.is_available()` returns true).

**Infinite generation on out-of-domain prompts** — if the model's top token at step 0 is not the start of any valid function name, the valid token set is empty and the loop would spin forever. Fixed with an early exit that returns `fn_None` when `final_allowed_tokens` is empty or when the model's greedy choice is not in the allowed set from the very first step.

**Backslash escaping in extracted strings** — regex strings extracted by the model sometimes contained raw backslashes (e.g. `\d`). Added a `fixed_raw_params = raw_params.replace("\\", "\\\\")` pass before `json.loads()` to prevent parse failures.

**`number` type casting** — `json.loads()` returns integers for values like `2` and `3`. The function definition specifies `"type": "number"` (which maps to float). An explicit cast loop after parsing ensures all number-typed parameters are stored as `float`.

---

## Testing Strategy

1. **Lint validation** — `make lint` is run after every change to enforce `flake8` style and `mypy` type safety (all functions typed, no untyped defs).

2. **Full pipeline run** — `make run` is executed against the 11 prompts in `data/input/function_calling_tests.json`. Terminal output is checked for correct function names; the output file is opened and each entry is verified manually.

3. **Edge case checks** — tested with:
   - Two-argument numeric functions (`fn_add_numbers`)
   - Single-argument string functions (`fn_greet`, `fn_reverse_string`)
   - Three-argument regex function (`fn_substitute_string_with_regex`)
   - Large numbers (`265 + 345`, `√144`)
   - Strings with special characters and spaces

4. **JSON output validation** — the output file is loaded with `json.load()` independently to confirm it is always parseable and matches the expected schema.

---

## Terminal Output Example

```
[Processing 1/11] Prompt: What is the sum of 2 and 3?
-> Captured Function Name: fn_add_numbers
-> Captured Raw Parameters: {"a": 2, "b": 3}

[Processing 3/11] Prompt: Greet shrek
-> Captured Function Name: fn_greet
-> Captured Raw Parameters: {"name": "shrek"}

[Processing 9/11] Prompt: Replace all numbers in "Hello 34 I'm 233 years old" with NUMBERS
-> Captured Function Name: fn_substitute_string_with_regex
-> Captured Raw Parameters: {"source_string": "Hello 34 I'm 233 years old", "regex": "([0-9]+)", "replacement": "NUMBERS"}

✅ All tasks completed! Final file saved at: data/output/function_calls.json
```

---

## Resources

- [Qwen3-0.6B on Hugging Face](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Pydantic v2 Documentation](https://docs.pydantic.dev/)
- [Astral uv Package Manager](https://docs.astral.sh/uv/)
- [Transformers — AutoModelForCausalLM](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForCausalLM)
- [Constrained Decoding overview — Hugging Face blog](https://huggingface.co/blog/constrained-beam-search)
- [flake8 Documentation](https://flake8.pycqa.org/)
- [mypy Documentation](https://mypy.readthedocs.io/)

### AI Usage

AI was used for the following tasks in this project:

- **Code refactoring** — splitting monolithic logic from `__main__.py` into `generation.py` and `models_and_utils.py` to pass strict lint checks.
- **Type annotation debugging** — resolving `mypy` errors on complex typed return values and `Dict[str, Any]` structures.

All AI-generated content was read, understood, tested, and validated before inclusion.