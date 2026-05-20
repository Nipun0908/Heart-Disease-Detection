
# SECTION 1: IMPORT LIBRARIES

import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns



# SECTION 2: DATA PREPARATION

disease_df = pd.read_csv("framingham.csv")
disease_df.drop(columns=['education'], inplace=True, axis=1)
disease_df.rename(columns={'male': 'Sex_male'}, inplace=True)

disease_df.dropna(axis=0, inplace=True)
print(disease_df)

print(disease_df.TenYearCHD.value_counts())


# SECTION 3: TRAIN/TEST SPLIT

from sklearn.model_selection import train_test_split

X = np.asarray(disease_df[['age', 'Sex_male', 'cigsPerDay',
                            'totChol', 'sysBP', 'glucose']])
y = np.asarray(disease_df['TenYearCHD'])

X = preprocessing.StandardScaler().fit(X).transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=4)

print('Train set:', X_train.shape, y_train.shape)
print('Test set:', X_test.shape, y_test.shape)


# SECTION 4: EXPLORATORY DATA ANALYSIS (EDA)

# 4.1 Count plot - patients with/without heart disease
plt.figure(figsize=(7, 5))
sns.countplot(x='TenYearCHD', data=disease_df, palette="BuGn_r")
plt.show()

# 4.2 Line plot of TenYearCHD column
laste = disease_df['TenYearCHD'].plot()
plt.show(laste)


# SECTION 5: LOGISTIC REGRESSION MODEL TRAINING


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)


# SECTION 6: MODEL EVALUATION


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Print accuracy
print('Accuracy of the model is =', accuracy_score(y_test, y_pred))

# Print classification report
print('Details for confusion matrix:')
print(classification_report(y_test, y_pred))

# Plot confusion matrix as heatmap
cm = confusion_matrix(y_test, y_pred)
conf_matrix = pd.DataFrame(data=cm,
                            columns=['Predicted:0', 'Predicted:1'],
                            index=['Actual:0', 'Actual:1'])

plt.figure(figsize=(8, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Greens")
plt.show()
