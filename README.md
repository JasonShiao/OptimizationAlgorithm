# OptimizationAlgorithm
Implement and test some optimization algorithms

## Prerequisite
* [https://python-poetry.org/docs/#installation](Poetry) Python package manager

## Run examples
```
poetry install

poetry run run_aco_example -h # See input options
poetry run run_aco_example --mode 'cycle' --problem 'gr17'

poetry run run_sma_example -h # See input options
poetry run run_sma_example # default Ackley function
poetry run run_sma_example --bf Schwefel --dim 10'
```


## Build the package
```
poetry install
poetry build
```

## Test 
```
# Test all
poetry run pytest
# Select test folder
poetry run pytest -- {folder to test}
# Print stdout
poetry run pytest -s -- {folder to test}
```
