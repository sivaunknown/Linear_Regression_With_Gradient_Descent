# Linear_Regression_With_Gradient_Descent
Gradient Descent Impleamentation

Here is a README.md file for the provided Python code.

-----

# Simple Linear Regression using Gradient Descent

This repository contains a Python script that demonstrates **Simple Linear Regression** using the **Gradient Descent** algorithm. The script trains a model on a dataset of years of experience and corresponding salaries to predict salary based on experience.

## Description

The `SLR_Gradient_Descent` class implements the core logic for a simple linear regression model. It uses gradient descent to iteratively find the best-fit line by minimizing the squared error between the predicted and actual values.

The key components of the script are:

  * **`SLR_Gradient_Descent` Class**: A class that encapsulates the data, learning rate, and the methods for training and plotting the model.
  * **`gradient_desc` Method**: This method performs one iteration of gradient descent. It calculates the gradients for the slope ($m$) and y-intercept ($b$) based on the current model parameters and updates them using the specified `learning_rate`.
  * **`plot_graph` Method**: This method visualizes the results. It displays the original data points as a scatter plot and overlays the final regression line found by the model.

## How it Works

The script follows these steps:

1.  **Data Loading**: The model loads a dataset named `Salary_dataset.csv`.
2.  **Initialization**: It initializes the model's parameters, `m` (slope) and `b` (y-intercept), to zero.
3.  **Training**: It runs a loop for 300 iterations, calling the `gradient_desc` method in each iteration to update `m` and `b`. The `learning_rate` of `0.01` controls the step size of each update.
4.  **Prediction & Visualization**: After training, the script uses the final `m` and `b` values to plot the best-fit line over the dataset, showing the model's prediction.

## Requirements

  * `pandas`: For data manipulation and loading the CSV file.
  * `matplotlib`: For plotting the results.

You can install these libraries using `pip`:

```bash
pip install pandas matplotlib
```

## Dataset

The script expects a CSV file named `Salary_dataset.csv` with at least two columns: `YearsExperience` and `Salary`.

An example of the dataset structure:

| YearsExperience | Salary |
|:---:|:---:|
| 1.1 | 39343.0 |
| 1.3 | 46205.0 |
| 1.5 | 37731.0 |
| ... | ... |

## Usage

1.  Place your `Salary_dataset.csv` file in the same directory as the Python script.
2.  Run the script from your terminal:

<!-- end list -->

```bash
python your_script_name.py
```

3.  A plot will be displayed showing the data and the linear regression line.

-----
