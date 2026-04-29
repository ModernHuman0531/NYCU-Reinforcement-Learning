PYTHON-VERSION=3.10.20

all: check-version setup-tool install
check-version:
	@which poetry > /dev/null || (echo "Poetry is not installed. Please install Poetry to proceed." && exit 1)
	@which pyenv > /dev/null || (echo "Pyenv is not installed. Please install Pyenv to proceed." && exit 1)

setup-tool: check-version
	@pyenv install $(PYTHON-VERSION) --skip-existing && (echo "Installation of Python $(PYTHON-VERSION) successfully") || (echo "Python $(PYTHON-VERSION) is already existed.")
	@pyenv local $(PYTHON-VERSION) && (echo "Set the project python version to $(PYTHON-VERSION)")
	@poetry config virtualenvs.in-project true && (echo "Configured Poetry to create virtual environments in the project directory.")
	@poetry env use $(PYTHON-VERSION) && (echo "Poetry is now using Python $(PYTHON-VERSION) for the virtual environments.")

install: setup-tool
	@poetry install && (echo "Dependencies installed successfully." || echo "Failed to install dsependencies.")

run: 
	@echo "To run the project, use the following command: poetry run python main.py"

clean:
	@rm -rf .venv 
	@poetry cache clear . --all
