# ML-Impute

### A python package for synthetic data generation using single and multiple imputation.

<div align="center" style="display: flex; justify-content: center;">

<a href="https://pypi.python.org/pypi/">
<img src ="https://img.shields.io/badge/python-3.x-blue.svg?style=for-the-badge" alt="Python version" /></a>

<!-- Build status -->
<a href="https://pypi.org/project/ml-impute">
<img src ="https://img.shields.io/pypi/v/ml-impute?style=for-the-badge" alt="PyPi version"/></a>

<!-- Test coverage -->
<!--
<a href="https://coveralls.io/">
<img src ="https://img.shields.io/codecov/c/gh/JoshWeiner/ml-impute.svg?style=for-the-badge" alt="Coverage Status"/></a>
-->

<a href="https://opensource.org/licenses/MIT">
<img src ="https://img.shields.io/:license-mit-ff69b4.svg?style=for-the-badge" alt="license" /></a>

</div>

Ml-Impute is a library for generating synthetic data for null-value imputation, notably with the ability to handle mixed datatypes. This package is based off of the research of [Audigier, Husson, and Josse](https://arxiv.org/pdf/1301.4797.pdf) and their method of iterative factor analysis for singular data imputation. <br>
The goal of this package is to: <br>
**(a)** provide an open source package for use of this method in Python for the first time, and; <br>
**(b)** to provide an efficient parallelization of the algorithm when extending it to both single and multiple imputation.

> Note: I am currently a university student and may not have the time to continue to release updates and changes as fast as some other packages might. In the spirit of open-source code, please feel free to add pull requests or open a new issue if you have bug fixes or improvements. Thank you for your understanding and for your contributions.
<hr>

## Table of Contents
- [Table of Contents](#table-of-contents)
- [Installation](#installation)
- [Usage](#usage)
- [Example](#example)
- [License](#license)

<hr>

## Installation

ML-Impute is currently available on PyPi.

**Unix/Mac OS/Windows**
```
pip install ml-impute
```
<hr>

## Usage
Currently, ML-Impute can handle both single and multiple imputation.

To follow a demonstration of both methods, proceed to the <a href="#Example">Example</a> Section. 

The following subsections provide an overview into each method along with their usage information.

To use the package post-installation via pip, instantiate the following object as follows:
```
from mpute import generator

gen = generator.Generator()
```

> #### **Generator.generate**(self, dataframe, encode_cols, exclude_cols, max_iter, tol, explained_var, method, n_versions, noise)
| Parameter | Description |
| :--- | :--- |
| dataframe | (__*required*__) Pandas dataframe object |
| encode_cols | (*optional*, default=[]) Categorical columns to be encoded. <br> By default, ml-impute will encode all columns with *object* or *category* dtypes. However, many datasets contain numerical categorical data (ex/ Likert scales, classification types, etc.) that should be encoded. |
| exclude_cols | (*optional*, default=[]) Categorical columns to be excluded from encoding and/or imputation. <br> On occastion, datasets will contain unique non-ordinal data (such as unique IDs) that, if encoded, will lead to large increases in memory usage and runtime. These columns should be excluded. |
| max_iter | (*optional*, default=1000) The maximum number of iterations of imputation before exit. |
| tol | (*optional*, default=1e-4) Tolerance bound for convergence. <br>If Frobenius norm relative error is < tol before max_iter is reached, exit.|
| explained_var | (*optional*, default=0.95) Percentage of the total variance kept when reconstructing the dataframe after performing Singular Value Decomposition. |
| method | (*optional*, default="single") Specification for use of single or multiple imputation method. <br> **Possible values**: ["single", "multiple"] |
| n_versions | (*optional*, default=20)  If performing multiple imputation, the number of generated dataframes. <br> If performing singular imputation, n_versions=1|
| noise | (*optional*, default="gaussian") If performing multiple impuation, specify the type of noise added to each generated dataset to create variation. Gaussian noise is centered around 0 with a standard deviation of 0.1. <br> If performing singular imputation, noise=None |
| engine | (*optional*, default="default") For either singular or multiple imputation, choose the engine through which the SVD is calculated. <br> **Possible values**: ["default", "dask"]<br>*"default"* utilizes the JAX numpy library for efficient SVD calculation and multiprocessing, and is recommended for speed. <br> *"dask"* creates a dask distributed scheduler which is used to compute the SVD. Given that this is an iterative method, this is recommended only when working with very large datasets. |

| Method | Return Value |
| :--- | :--- |
| "single" | **imputed_df**: a copy of the dataframe argument with synthetic data imputed for all null values |
| "multiple" | **df_dict**: a dictionary containing each of the n_versions of generated datasets with variable synthetic data. <br> keys: [0, n_versions) <br> values: [dataframes]|

<hr>

### **Single Imputation**
Single imputation works with the following line:
```
imputed_df = gen.generate(dataframe)
```
### **Multiple Imputation**
Multiple imputation is as simple as the following:
```
imputed_dfs = gen.generate(dataframe method="multiple")
```

<hr>

## Example

For the following example, we will use the titanic example-dataset available in [sklearn.datasets openml](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_openml.html#sklearn.datasets.fetch_openml).

Build the titanic dataset and create a Generator object as follows:
```
import pandas as pd
from mpute import generator
from sklearn import datasets

titanic, target = datasets.fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)
titanic['survived'] = target

gen = generator.Generator()
```
### **Single Imputation**

```
imputed_df = gen.generate(titanic, exclude_cols=['name', 'cabin', 'ticket'])
```
> **Note**: 'name', 'cabin', and 'ticket' are excluded as they mainly contain unique identifiers, therefore unnecessary for imputation and if encoded, would result in a significant increase in memory usage. <br>
> It is possible to replace the cabin column with two columns such as 'deck' and 'position', as these may be a determinant of survival. However, this preprocessing would have to occur beforehand 
<hr>

### **Multiple Imputation**
Multiple imputation is as simple as the following:
```
imputed_dfs = gen.generate(titanic method="multiple")
```

That's all there is to it. Happy using!
<hr>

## License
ML-Impute is published under the MIT License. Please see the LICENSE file for more information.