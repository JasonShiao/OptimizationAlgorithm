# OptimizationAlgorithm
Implement and test some optimization algorithms

## Prerequisite
* [Poetry Python package manager](https://python-poetry.org/docs/#installation)

## Run examples
```
poetry install

poetry run run_aco_example -h # See input options
poetry run run_aco_example --mode 'cycle' --problem 'gr17'
poetry run run_aco_example --mode 'cycle' --problem 'st70'
poetry run run_aco_example --mode 'cycle' --problem 'att48'
poetry run run_aco_example --mode 'cycle' --problem 'gr202'
poetry run run_aco_example --mode 'cycle' --problem 'a280'

poetry run run_pso_example -h # See input options
poetry run run_pso_example --bf Ackley --dim 5

poetry run run_sma_example -h # See input options
poetry run run_sma_example --bf Ackley --dim 10
poetry run run_sma_example --bf Schwefel --dim 10
```
### Example TSP visualization result (st70)
 node            |  1
:-------------------------:|:-------------------------:
![st70 node](https://github.com/JasonShiao/OptimizationAlgorithm/blob/develop/docs/img/st70_node.png?raw=true)  |  ![st70 1](https://github.com/JasonShiao/OptimizationAlgorithm/blob/develop/docs/img/st70_1.png?raw=true)
 2            |  3
![st70 2](https://github.com/JasonShiao/OptimizationAlgorithm/blob/develop/docs/img/st70_2.png?raw=true)  |  ![st70 3](https://github.com/JasonShiao/OptimizationAlgorithm/blob/develop/docs/img/st70_3.png?raw=true)
 13            |  14
 ![st70 13](https://github.com/JasonShiao/OptimizationAlgorithm/blob/develop/docs/img/st70_13.png?raw=true)  |  ![st70 14](https://github.com/JasonShiao/OptimizationAlgorithm/blob/develop/docs/img/st70_14.png?raw=true)


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
