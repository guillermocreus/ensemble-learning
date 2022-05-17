# Ensemble learning - Guillermo Creus

## Report
The report of this task can be found in the folder ./documentation

## Data
Data for this task can be found in the folder ./data

The datasets are the following:

- Small: Wine
- Medium: Breast cancer Wisconsin
- Large: Mushroom

## Code
The code of this task can be found in ./source. 

The python file of my implementation of Random Forests can be found in RandomForest_df.py. 
It is a class and one should initialize it with the F and NT hyperparameters. After that, one 
can call the fit function, where the inputs X, y should be pandas dataframes. For predicting,
the input will be X, a pandas dataframe. Also, note that one of its attributes is the feature 
importances.

Regarding my implementation of Decision Forests, it can be found in DecisionForest_df.py. 
It is a class and one should initialize it with the F and NT hyperparameters (it accepts F = -1,
which means that it will choose F for every tree as a random integer between 1 and the number of 
features). After that, one can call the fit function, where the inputs X, y should be pandas 
dataframes. For predicting, the input will be X, a pandas dataframe. Also, note that one of its 
attributes is the feature importances.

Both implementations accept mixed datasets, i.e., ones with continuous and categorical variables.
In addition, they use the class found CART_df.py, which is my implementation of CART. The latter 
decision tree uses a helper class for the nodes that can be found in Node_G_df.py.

Lastly, it should be mentioned that the notebook used to extract all of the results is found in 
./source/main.ipynb and it is very straightforward to use. It prints all of the results and one 
can easily choose the dataset to analyze: small, medium or large.
