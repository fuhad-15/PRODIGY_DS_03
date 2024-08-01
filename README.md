# PRODIGY_DS_03

## Decision Tree Classifier for Customer Purchase Prediction

### Description

This repository contains code and analysis for building a decision tree classifier to predict whether a customer will purchase a product or service based on their demographic and behavioral data. The analysis uses the Bank Marketing dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing).

### Project Overview

The project includes:

- **Data Preparation:** Loading the dataset, cleaning data, and preparing features for the model.
- **Model Building:** Implementing a decision tree classifier to predict customer purchase behavior.
- **Model Evaluation:** Assessing the performance of the classifier using metrics like accuracy, precision, recall, and F1-score.
- **Visualization:** Visualizing the decision tree and feature importance.

### Getting Started

To build and evaluate the decision tree classifier, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/decision-tree-customer-purchase.git
   ```

2. **Navigate to the Project Directory:**
   ```bash
   cd decision-tree-customer-purchase
   ```

3. **Install Dependencies:**
   ```bash
   pip install pandas numpy scikit-learn matplotlib graphviz
   ```

4. **Download the Bank Marketing Dataset:**
   Download the dataset from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing) and place it in the `data/` directory as `bank_marketing.csv`.

5. **Run the Classifier Script:**
   ```bash
   python decision_tree_classifier.py
   ```

6. **View Results:** The script generates a decision tree visualization and saves performance metrics in the `results/` directory.


### Key Findings

- **Feature Importance:** Identified key features that influence whether a customer will purchase a product or service.
- **Model Performance:** Achieved an accuracy of X% with the decision tree classifier.
- **Insights:** Analyzed the most influential factors affecting customer purchase decisions.



### `decision_tree_classifier.py` Script

Here is a basic example of the `decision_tree_classifier.py` script:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
from sklearn import tree

# Load the dataset
data = pd.read_csv('data/bank_marketing.csv')

# Data Preparation
data.replace({'yes': 1, 'no': 0}, inplace=True)  # Convert categorical responses to numerical
data['education'].replace({'basic.4y': 0, 'basic.6y': 1, 'basic.9y': 2, 'high.school': 3, 'illiterate': 4, 'professional.course': 5, 'university.degree': 6, 'unknown': np.nan}, inplace=True)  # Convert education levels to numerical
data.dropna(inplace=True)  # Drop rows with missing values

# Define features and target
X = data.drop('y', axis=1)
y = data['y']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict and Evaluate the Model
y_pred = clf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('results/confusion_matrix.png')
plt.show()

# Visualize the Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=X.columns, class_names=['No', 'Yes'], filled=True)
plt.title('Decision Tree Visualization')
plt.savefig('results/decision_tree.png')
plt.show()
```

Replace `path_to_your_image/` with the path where your actual images will be stored.

### Full README Example

Here’s how your full README file might look with the added section:

```markdown
# Decision Tree Classifier for Customer Purchase Prediction

This repository contains code and analysis for building a decision tree classifier to predict whether a customer will purchase a product or service based on their demographic and behavioral data. The analysis uses the Bank Marketing dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing).

## Project Overview

The project includes:

- **Data Preparation:** Loading the dataset, cleaning data, and preparing features for the model.
- **Model Building:** Implementing a decision tree classifier to predict customer purchase behavior.
- **Model Evaluation:** Assessing the performance of the classifier using metrics like accuracy, precision, recall, and F1-score.
- **Visualization:** Visualizing the decision tree and feature importance.

## Getting Started

To build and evaluate the decision tree classifier, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/decision-tree-customer-purchase.git
   ```

2. **Navigate to the Project Directory:**
   ```bash
   cd decision-tree-customer-purchase
   ```

3. **Install Dependencies:**
   ```bash
   pip install pandas numpy scikit-learn matplotlib graphviz
   ```

4. **Download the Bank Marketing Dataset:**
   Download the dataset from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing) and place it in the `data/` directory as `bank_marketing.csv`.

5. **Run the Classifier Script:**
   ```bash
   python decision_tree_classifier.py
   ```

6. **View Results:** The script generates a decision tree visualization and saves performance metrics in the `results/` directory.


## Key Findings

- **Feature Importance:** Identified key features that influence whether a customer will purchase a product or service.
- **Model Performance:** Achieved an accuracy of X% with the decision tree classifier.
- **Insights:** Analyzed the most influential factors affecting customer purchase decisions.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.


## `decision_tree_classifier.py` Script

Here is a basic example of the `decision_tree_classifier.py` script:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
from sklearn import tree

# Load the dataset
data = pd.read_csv('data/bank_marketing.csv')

# Data Preparation
data.replace({'yes': 1, 'no': 0}, inplace=True)  # Convert categorical responses to numerical
data['education'].replace({'basic.4y': 0, 'basic.6y': 1, 'basic.9y': 2, 'high.school': 3, 'illiterate': 4, 'professional.course': 5, 'university.degree': 6, 'unknown': np.nan}, inplace=True)  # Convert education levels to numerical
data.dropna(inplace=True)  # Drop rows with missing values

# Define features and target
X = data.drop('y', axis=1)
y = data['y']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict and Evaluate the Model
y_pred = clf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('
