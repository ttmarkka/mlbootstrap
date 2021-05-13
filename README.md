# BootstrapEstimator: bootstrap estimates for machine learning

The bootstrap method is a commonly used resampling technique for evaluating the variability of a model when applied to new data [Bootstrapping (statistics)](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)).

This Python class aims to make this process automatic, while providing a **method for generating a plot**. Intended to be used with [sklearn](https://scikit-learn.org/stable/).

Includes functionality appropriate for **times series** data, where splitting data into training and testin is done in a time-ordered manner.

![image](/image/LR.png)

## Installation
This project is not packaged. Simply copy the file **BootstrapEstimator.py** to your working directory and use

```python
# Class
from BootstrapEstimator import BootstrapEstimator
# Just the train/test splitter function for time series
from BootstraEstimator import train_test_split_ts
```

## Dependencies
Uses
- pandas
- numpy
- sklearn
- matplotlib
- seaborn

Currently tested for versions 1.1.3, 1.19.2, 0.23.2, 3.3.2 and 0.11.0, respectively.

## Usage
The use and conventions try to follow those in [sklearn](https://scikit-learn.org/stable/) as closely as possible. As an example of application to regression, for given data and target values X and y, one can for example choose to use
*sklearn.linear_model.LinearRegression* and *sklearn.metrics.mean_squared_error*, which allow to produce the above plot as:

```python
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import mean_squared_error as mse

# Create an instance of the class
BE = BootstrapEstimator(LR())

# Fit bootstrap models and calculate the MSE for each replica
BE.fit_calculate(X, y, mse, n = 10000, test_size = 0.1)

BE.plot(bins = 60)
```
Please see the Jupyter notebook for more information and examples.

## version
0.0.2
