**__Healthcare Capstone Project - Predicting Diabetes__**

_**Problem Statement**_

The objective of this project is to predict whether a patient has diabetes or not based on certain diagnostic measurements. The dataset consists of several medical predictor variables and one target variable, Outcome.

_**Dataset Description**_

The dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases and includes data on females of Pima Indian heritage who are at least 21 years old. The following variables are included in the dataset:

Pregnancies: Number of times pregnant \n
Glucose: Plasma glucose concentration 2 hours in an oral glucose tolerance test
BloodPressure: Diastolic blood pressure (mm Hg)
SkinThickness: Triceps skin fold thickness (mm)
Insulin: 2-Hour serum insulin (mu U/ml)
BMI: Body mass index (weight in kg/(height in m)^2)
DiabetesPedigreeFunction: Diabetes pedigree function
Age: Age (years)
Outcome: Class variable (0 or 1)

_**Approach**_

The following steps were taken in the analysis:

1. Descriptive analysis was performed to understand the variables and corresponding values. Missing values were identified in columns with minimum values of 0, including Glucose, BloodPressure, SkinThickness, Insulin, and BMI. Missing values were treated using appropriate methods.
2. Variables were visually explored using histograms to understand the distribution and treat missing values accordingly.
3. A count plot was created to describe the data types and count of variables.
4. The data's balance was checked by plotting the outcomes' count by their value. The data was found to be imbalanced, and future courses of action were planned accordingly.
5. Scatter charts were created between the pair of variables to understand the relationships between them.
6. Correlation analysis was performed visually using a heat map.
7. Strategies were devised for model building, and the right validation framework was decided upon. Cross-validation was found to be useful in this scenario.
8. An appropriate classification algorithm was applied to build a model. Various models were compared with the results from KNN.
9. A classification report was created by analyzing sensitivity, specificity, AUC (ROC curve), and other parameters. The values of these parameters were settled upon after thorough analysis.
10. A dashboard was created in Tableau using appropriate chart types and metrics useful for the business. The dashboard included a pie chart to describe the diabetic/non-diabetic population, scatter charts between relevant variables, histogram/frequency charts to analyze the distribution of the data, a heatmap of correlation analysis among relevant variables, and a bubble chart analyzing different variables for different age brackets.

_**Conclusion**_

In conclusion, this project was successful in building a model to accurately predict whether the patients in the dataset have diabetes or not. The findings from the exploratory analysis and the insights gained through the model building process were valuable in providing insights into the problem. The dashboard created in Tableau is a useful tool for visualizing the data and communicating the findings to stakeholders.
