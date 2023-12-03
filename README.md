# OptimizationAlgorithm
Implement and test some optimization algorithms

## Run examples
```
poetry install
poetry run run_example -h # See input options
poetry run run_example --mode 'cycle' --problem 'gr17'
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
