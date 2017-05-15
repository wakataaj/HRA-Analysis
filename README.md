# HRA-Analysis

This was a simple data analysis, manipulation, and predictive modeling of the Human Resources Analytics data set from kaggle.com. 

I took the following steps in my initial data analysis:
  - Preliminary data exploration (df.head(), df.describe(),df.info(), etc)
  - Examining various reltionships between certain independent variables and the dependent variable "left" that we will try to predict later.
  -I ran code to creat a correlation heat map to examine which dependent varibales were especially important to whether or not an employee left or not.
 
Luckily there weren't any sort of outliers or missing variables to handle, so I moved onto the data transformation step.

In my effort to transform the dataset into a form that I could effectively run a few classifier models on, I took the following steps:
  - I converted 'statisfaction_level', 'average_monthly_hours', and 'last_evaluation' into the numerical categories of 0,1,2 by dividing the entire range of each variable into three equal ranges, this allowed me to convert them easily into numerical categories that still held their significance.
  - I turned the 'sales' and 'salary' catergorical variables into numerical variables by assigning each unique value to a number.
  
Having been satisfied with the dataset, I moved onto the preliminary tasks for predictive modeling.

I split the data into training and tests sets and ran a grid search with multiple variations of hyperparameters respective to each model in order to get the best parameters for each model.

After running the models, I found that the K Nearest Neighbors Classifier and the Support Vector Classiffier performed the best by not only just scoring the model, but by also looking at the classification report which I feel is a great indicator of how well a classifier performs on a binary classification target variable. Both models had high Precision, Recall, and consequentially f-1 scores which further confirmed my confidence on either models for unseen data given the dataset manipulation steps I took above.
