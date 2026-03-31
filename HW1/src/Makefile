# Set the global variable for the compiler
SPECIFIED-PYTHON=3.10

all: check-tool setup install
# To ensure poetry and pyenv are installed.
check-tool:
	@which poetry > /dev/null || (echo "Poetry is not installed. Please install Poetry to proceed." && exit1)
	@which pyenv > /dev/null || (echo "Pyenv is not installed. Please install Pyenv to proceed." && exit 1)

# setup the python environment using pyenv and poetry
setup: check-tool
	@pyenv install $(SPECIFIED-PYTHON) --skip-existing && (echo "Installation of Python $(SPECIFIED-PYTHON) successful." || echo "Python $(SPECIFIED-PYTHON) is already installed.")
	@pyenv local $(SPECIFIED-PYTHON) && (echo "Set the local python version to $(SPECIFIED-PYTHON)")
	@poetry config virtualenvs.in-project true && (echo "Configured Poetry to create virtual environments in the project directory.")
	@poetry env use $(SPECIFIED-PYTHON) && (echo "Poetry is now using Python $(SPECIFIED-PYTHON) for the virtual environment.")

# Install the dependencies using portry
install: setup
	@poetry install && (echo "Dependencies installed successfully." || echo "Failed to install dependencies.")

# Hint for the user to run the project
run: install
	@echo "To run the project, use the following command: poetry run python reinforce.py"

# Clean up the environment and remove the virtual environment and cache
clean:
	@echo "Removing virtual environment"
	@rm -rf .venv

	@echo "Removing Python cache files"
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete

	@echo "Removing tool caches ..."
	@rm -rf .pytest_cache .mypy_cache .cache

	@echo "Clean up completed."
