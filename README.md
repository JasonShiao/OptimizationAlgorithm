# OptimizationAlgorithm
Implement and test some optimization algorithms

## Prerequisite
* [Poetry Python package manager](https://python-poetry.org/docs/#installation)

## Run examples
```
poetry install

poetry run run_aco_example -h # See input options
poetry run run_aco_example --mode 'cycle' --problem 'gr17'
poetry run run_aco_example --mode 'cycle' --problem 'gr202'
poetry run run_aco_example --mode 'cycle' --problem 'a280'
poetry run run_aco_example --mode 'cycle' --problem 'att48'

poetry run run_sma_example -h # See input options
poetry run run_sma_example --bf Ackley --dim 10
poetry run run_sma_example --bf Schwefel --dim 10

poetry run run_sma_example -h # See input options
poetry run run_sma_example --bf Ackley --dim 10
poetry run run_sma_example --bf Schwefel --dim 10
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
