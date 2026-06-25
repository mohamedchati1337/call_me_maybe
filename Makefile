func_def ?= data/input/functions_definition.json
input    ?= data/input/function_calling_tests.json
output   ?= data/output/function_calls.json


install:
	pip install uv
	uv sync 




run:
	uv run python -m src --functions_definition  $(func_def) --input  $(input) --output $(output) 

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -exec rm -f {} +
	find . -type d -name ".mypy_cache" -exec rm -r {} +
	rm -rf .python-version


debug:
	uv run python -m pdb -m src --functions_definition  $(func_def) --input  $(input) --output $(output) 




lint:
	uv run flake8 src
	uv run mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs


lint-strict:
	uv run flake8 src
	uv run mypy src --strict