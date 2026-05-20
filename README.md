# Heart-Disease-Detection
Machine learning become very helpful in healthcare for predicting conditions like heart disease. We use linear regression which helps to predict the likelihood of something happening like whether a person has heart disease based on input features.This setup handles the ingestion of patient diagnostic data (like age, cholesterol, maximum heart rate, blood pressure, etc.), data cleaning, standardization, feature-to-target evaluation, and classification.
## Theoretical Overview
Logistic Regression mapping relies on the standard Sigmoid mathematical transformation function:

$$P(Y=1|X) = \frac{1}{1 + e^{-z}}$$

Where $z = \beta_0 + \beta_1x_1 + \beta_2x_2 + \dots + \beta_nx_n$. This yields a probabilistic output boundary between $0$ and $1$. Predictions are categorized across a classification threshold boundary (defaulting to $0.5$):
* **Target = 0**: Lower indication of narrow blood vessels (lower risk).
* **Target = 1**: Narrowing of blood vessels greater than 50% threshold (elevated clinical risk).
## Requirements
Install following libreries
* pandas 
* numpy 
* scikit-learn 
* matplotlib 
* seaborn
## Setup & Running Instructions
1.	Open this repository locally

2. Acquire Dataset:
   * Download the popular UCI Heart Disease compilation dataset file `heart-.csv`.
   * Place the `heart.csv` file directly inside the repository root directory.

3. Install Dependencies:

   requirements.txt
