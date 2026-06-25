# create makefile for building virtual environment and run scripts by using uv

# Set environment variables
PYTHON-VERSION:=3.10
TARGET_FILE:=train_dpo_starter.py

# .PHONY the targets that are not files
.PHONY: all check-uv-installed get-python-version init modify-depend run clean

all: check-uv-installed get-python-version init
check-uv-installed:
	@command -v uv > /dev/null || (echo "uv is not installed. Please install uv first." && exit 1)
get-python-version:
	@uv python install $(PYTHON-VERSION) && echo "Python $(PYTHON-VERSION) installed"
init:
	@test -f pyproject.toml || (uv init --python $(PYTHON-VERSION) && echo "Virtual environment initialized")
# Run this manually after editing pyproject.toml.
modify-depend:
	@uv lock && echo "Synchronize dependencies with the lock file and pyproject.toml"
	@uv sync && echo "Modify dependencies in the virtual environment according to the lock file and pyproject.toml"
run:
	@uv run python ${TARGET_FILE} && echo "Run ${TARGET_FILE} in the virtual environment"

clean:
	@rm -rf .venv && echo "Virtual environment removed"
