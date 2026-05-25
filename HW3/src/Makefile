PYTHON-VERSION=3.10.20

all: check-version setup-tool install

check-version:
	@which poetry > /dev/null || (echo "Poetry is not installed. Please install Poetry to continue." && exit 1)
	@which pyenv > /dev/null || (echo "Pyenv is not installed. Please install Pyenv to continue." && exit 1)

setup-tool:check-version
	@pyenv install $(PYTHON-VERSION) --skip-existing && (echo "Installation of Python $(PYTHON-VERSION) successful." || (echo "Failed to install Python $(PYTHON-VERSION)." && exit 1))
	@pyenv local $(PYTHON-VERSION) && (echo "Set the project Python version to $(PYTHON-VERSION).")
	@poetry config virtualenvs.in-project true && (echo "Configured Poetry to create virtual environments in the project directory.")
	@poetry env use $(PYTHON-VERSION) && (echo "Poetry is now using Python $(PYTHON-VERSION) for this project.")

install:setup-tool
	@poetry install && (echo "Project dependencies installed successfully." || (echo "Failed to install project dependencies." && exit 1))

run:
	@echo "To run the project, use: poetry run python <your_script.py>"

clean:
	@rm -rf .venv
	@poetry cache clear . --all
	@echo "Cleaned up the project environment."