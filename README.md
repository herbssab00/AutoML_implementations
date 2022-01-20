# Automated Machine Learning

This project contains implementations and tests with automated machine learning algorithms.

The `requirements.txt` file contains the necessary dependencies. The project was created using Python 3.10 and anaconda.
Please install the dependencies using `conda create --name <env> --file requirements.txt`

---
## Hyperband
The hyperband algorithm was proposed by Li et al. in 2017 and makes use of Successive Halfing to find an optimal model and
hyper-parameters. For further information please refer to the original paper in [1].

I used four different regression models (KNN, SVR, RF, Lasso) to implement the algorithm. In practice, this can be extended 
with other models and also for classification tasks.

### Files
- `hyperband.py`: contains the implementation of the hyperband algorithm
    - `hyperband_algorithm`: this function that needs to be called and is reponsible for the actual execution of the algorithm
    - `get_hyperparameter_configurations`: this function returns **n** models and hyper-parameter configurations
    - `run_then_return_val_loss`: the function fits a given model and returns the loss on an independent test set
    - `top_k`: returns the **k** best configurations
- `main.py`: runs the hyperband algorithm for a given dataset

### Data
The data was downloaded from the UCI repository [2] via the following link https://archive.ics.uci.edu/ml/datasets/Auto+MPG.

### Citation
[1] Lisha Li, Kevin Jamieson, Giulia DeSalvo, Afshin Rostamizadeh, and Ameet Talwalkar. 2017. Hyperband: a novel bandit-based approach to hyperparameter optimization. J. Mach. Learn. Res. 18, 1 (January 2017), 6765–6816. \
[2] Dua, D. and Graff, C. 2019. UCI Machine Learning Repository http://archive.ics.uci.edu/ml. Irvine, CA: University of California, School of Information and Computer Science. 