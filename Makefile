# Environment variables
PYTHON = python
PIP = pip
REMOVE = rm -rf
CREATE = mkdir
PRINT = @echo

# targets
## setup :		Setup the virtual environment.
.PHONY: setup

## install :		Install dependencies.
.PHONY: install

## update :		Update dependencies.
.PHONY: update

## experiments : 	Run experiments.
.PHONY: experiments

## clean :		Clean up.
.PHONY: clean

# recipes
setup: create_env 
install: install_requirements
update: update_requirements

# rules
create_env:
	$(PRINT) "Creating Virtual Environment..."
	$(PYTHON) -m venv ./.venv
	$(PRINT)

install_requirements:
	$(PRINT) "Installing Dependencies..."
	$(PIP) install -r ./requirements.txt
	$(PRINT) ""

update_requirements:
	$(PRINT) "Updating Dependencies..."
	$(PIP) freeze > ./requirements.txt
	$(PRINT) ""

experiments:
	$(PRINT) "Running experiments..."
	$(PYTHON) ./src/experiments.py
	$(PRINT) ""

clean:
	$(PRINT) "Cleaning weights"
	$(REMOVE) ./weights
	$(CREATE) weights
	$(PRINT) "Cleaning generated samples"
	$(REMOVE) ./samples
	$(CREATE) samples
	$(PRINT) ""

