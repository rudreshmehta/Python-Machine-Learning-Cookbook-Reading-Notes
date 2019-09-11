- [Machine Learning with Python cookcook](#machine-learning-with-python-cookcook)
  - [1. vectors, matrices and arrays](#1-vectors-matrices-and-arrays)
  - [2. Loading data](#2-loading-data)
  - [3. Data wrangling](#3-data-wrangling)
  - [4. Handling numeric data](#4-handling-numeric-data)
  - [5. Handling categorical data](#5-handling-categorical-data)
  - [6. Handling Text](#6-handling-text)
  - [7. Handling dates and times](#7-handling-dates-and-times)
  - [8. Handling Images](#8-handling-images)
  - [9. Dimensionality reduction using feature extraction](#9-dimensionality-reduction-using-feature-extraction)
  - [10. Dimensionality reduction suing feature selection](#10-dimensionality-reduction-suing-feature-selection)
  - [11. Model evaluation](#11-model-evaluation)
    - [11.1 cross-validating models](#111-cross-validating-models)
    - [11.2 creating a baseline regression model](#112-creating-a-baseline-regression-model)
    - [11.3 creating a baseline classification model](#113-creating-a-baseline-classification-model)
    - [11.4 evaluating binary classifier predictions](#114-evaluating-binary-classifier-predictions)
    - [11.5 evaluating binary classifier thresholds](#115-evaluating-binary-classifier-thresholds)
    - [11.6 evaluating multiclass classifier predictions](#116-evaluating-multiclass-classifier-predictions)
    - [11.7 visualizing a classifier's performance](#117-visualizing-a-classifiers-performance)
    - [11.8 evaluating regression models](#118-evaluating-regression-models)
    - [11.9 evaluating clustering models](#119-evaluating-clustering-models)
    - [11.10 creatiwng a custom evaluation metric](#1110-creatiwng-a-custom-evaluation-metric)
    - [11.11 visualizing the effect of training set size](#1111-visualizing-the-effect-of-training-set-size)
    - [11.12 creating a text report of evalution metrics](#1112-creating-a-text-report-of-evalution-metrics)
    - [11.13 visualizing the effect of hyperparameter values](#1113-visualizing-the-effect-of-hyperparameter-values)
  - [12. Model selection](#12-model-selection)
    - [12.1 selecting best models using exhaustive search](#121-selecting-best-models-using-exhaustive-search)
    - [12.2 selecting best models using randomized search](#122-selecting-best-models-using-randomized-search)
    - [12.3 selecting best models from multiple learning algorithms](#123-selecting-best-models-from-multiple-learning-algorithms)
    - [12.4 selecting best models when preprocessing](#124-selecting-best-models-when-preprocessing)
    - [12.5 speeding up model selection with parallelization](#125-speeding-up-model-selection-with-parallelization)
    - [12.7 evaluating performance after model selection](#127-evaluating-performance-after-model-selection)
  - [13. Linear regression](#13-linear-regression)
    - [13.1 fitting a line](#131-fitting-a-line)
    - [13.2 handling interactive effects](#132-handling-interactive-effects)
    - [13.2 fitting a nonlinear relationship](#132-fitting-a-nonlinear-relationship)
    - [13.4 reading variance with regularization](#134-reading-variance-with-regularization)
    - [13.5 reducing features with Lasso regression](#135-reducing-features-with-lasso-regression)
  - [15. K-Nearest Neighbors](#15-k-nearest-neighbors)
    - [15.1 finding an observation's nearest neighbors](#151-finding-an-observations-nearest-neighbors)
    - [15.2 creating K-Nearest neighbor classifier](#152-creating-k-nearest-neighbor-classifier)
    - [15.3 identifying the best neighborhood size](#153-identifying-the-best-neighborhood-size)
    - [15.3 creating a redius-based nearest neighbor classifier](#153-creating-a-redius-based-nearest-neighbor-classifier)

# Machine Learning with Python cookcook

## 1. vectors, matrices and arrays

## 2. Loading data

## 3. Data wrangling

## 4. Handling numeric data

## 5. Handling categorical data

## 6. Handling Text

## 7. Handling dates and times

## 8. Handling Images

## 9. Dimensionality reduction using feature extraction

## 10. Dimensionality reduction suing feature selection

## 11. Model evaluation

### 11.1 cross-validating models

```python
from sklearn.model_selection import Kfold, cross_val_score

cv_results = cross_val_score(estimator, features, target, cv=Kold(n_splits=n), scoring='accuracy')
```

### 11.2 creating a baseline regression model

```python
from sklearn.dummy import DummyRegressor

dummy = DummyRegressor(strategy='mean')
# or
dummy = DummyRegressor(strategy='constant', constant=20)
```

### 11.3 creating a baseline classification model

```python
from sklearn.dummy import DummyClassifier

dummy = DummyClassifier(strategy='uniform', random_state=1)
```

### 11.4 evaluating binary classifier predictions

$$
\text {Accuracy}=\frac{T P+T N}{T P+T N+F P+F N}
$$

$$
\text { Precision }=\frac{T P}{T P+F P}
$$

$$
\text { Recall }=\frac{T P}{T P+F N}
$$

$$
F_{1}=2 \times \frac{\text { Precision } \times \text { Recall }}{\text { Precision }+\text { Recall }}
$$

### 11.5 evaluating binary classifier thresholds

The $Receiving \, Operating \, Characterisitc (ROC)$ curve is a commom method for evaluating the quality of a binary classifier, ROC compares the presence of true positives(TP) and false positives(FP) at every probability threshold. ($i.e. the probability at which an observation is predicted to be a class$)

$$
\mathrm{TPR}=\frac{\text { True Positives }}{\text { True Positives+False Negatives }}
$$

$$
\mathrm{FPR}=\frac{\text { False Positives }}{\text { False Positivest True Negatives }}
$$

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score

logit = LogisticRegression()
logit.fit(features_train, target_train)

# function - predict_proba() returns the probability of the sample for each class in the model, where classes are ordered as they are in self.classes_
target_probabilities = logit.predict_proba(feature_test)[:, 1]

false_positive_rate, true_positive_rate, threshold = roc_curve(target_test, target_probabilities)

plt.plot(false_positive_rate, true_positive_rate)
```

### 11.6 evaluating multiclass classifier predictions

### 11.7 visualizing a classifier's performance

```python
from sklearn.metrics import confusion_matrix
import pandas as pd

matrix = confusion_matrix(target_test, target_predicted)
dataframe = pd.DataFrame(matrix, index=class_names, columns=class_names)
sns.heatmap(dataframe, annot=True, cbar=None, cmap="Blues")
```

### 11.8 evaluating regression models

use mean squared error(MSE)

$$
\mathrm{MSE}=\frac{1}{n} \sum_{i=1}^{n}\left(\widehat{y_{i}}-y_{i}\right)^{2}
$$

$$
R^{2}=1-\frac{\sum_{i=1}^{n}\left(y_{i}-\hat{y}_{i}\right)^{2}}{\sum_{i=1}^{n}\left(y_{i}-\overline{y}\right)^{2}}
$$

### 11.9 evaluating clustering models

$$
s_{i}=\frac{b_{i}-a_{i}}{\max \left(a_{i}, b_{i}\right)}
$$

$$
\begin{array}{l}{\text { where } s_{i} \text { is the silhouette coefficient for observation } i, a_{i} \text { is the mean distance between }} \\ {i \text { and all observations of the same class, and } b_{i} \text { is the mean distance between } i \text { and all }} \\ {\text { observations from the closest cluster of a different class. The value returned by sil }} \\ {\text { houette }_{-} \text {score is the mean silhouette coefficient for all observations. Silhouette coef- }} \\ {\text { ficients range between }-1 \text { and } 1, \text { with } 1 \text { indicating dense, well-separated clusters. }}\end{array}
$$

```python
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

silhouette_score(features, target_predicted)
```

### 11.10 creatiwng a custom evaluation metric

### 11.11 visualizing the effect of training set size

```python
from sklearn.ensemble import RandomForestClassifier
from sklean.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(RandomForestClassifier(), features, target, cv=n, scoring='accuracy', train_sizes=np.linspace(0.01, 1, 50))

# Create means and standard deviations of training set scores
train_maen = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Create means and standard deviations of test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Draw lines
plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

# Draw bands
plt.fill_between(train_sizes, train_mean - train_std,
                 train_mean + train_std, color="#DDDDDD")
plt.fill_between(train_sizes, test_mean - test_std,
                 test_mean + test_std, color="#DDDDDD")
```

### 11.12 creating a text report of evalution metrics

```python
from sklearn.metrics import classification_report

print(classification_report(target_test, target_predicted, target_names=class_names))
```

### 11.13 visualizing the effect of hyperparameter values

```python
from sklearn.model_selection import validation_curve

param_range = np.arange(1, 250, 2)
train_scores, test_scores = validation_curve(estimator, features, target, param_name='n_estimators', param_range=param_range, cv=n, scoring='accuracy')

# Calculate mean and standard deviation for training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Calculate mean and standard deviation for test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
```

## 12. Model selection

### 12.1 selecting best models using exhaustive search

```python
import numpy as np
from sklearn import Linear_model
from sklearn.model_selection import GridSearchCV

logistic = linear_model.LogisticRegression()

# Create range of candidate penalty hyperparameter values
penalty = ['l1', 'l2']
# Create range of candidate regularization hyperparameter values
C = np.logspace(0, 4, 10)
# Create dictionary hyperparameter candidates
hyperparameters = dict(C=C, penalty=penalty)

gridsearch = GridSearchCV(logistic, hyperparameters, cv=5, verbose=0)
best_model = gridsearch.fit(features, target)
```

### 12.2 selecting best models using randomized search

```python
from scipy.stats import uniform
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV

logistic = LinearRegression()

# Create range of candidate regularization penalty hyperparameter values
penalty = ['l1', 'l2']
# Create distribution of candidate regularization hyperparameter values
C = uniform(loc=0, scale=4)
# Create hyperparameter options
hyperparameters = dict(C=C, penalty=penalty)

randomizedsearch = RandomizedSearchCV(logistic, hyperparameters, random_state=1, n_iter=100, cv=5, verbose=0)
best_model = randomizedsearch.fit(features, target)
```

### 12.3 selecting best models from multiple learning algorithms

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

pipe = Pipeline([('classifier', RandomForestClassifier())])

search_space = [{'classifier': [LogisiticRegression()], \
                'classifier__penalty': ['l1', 'l2'], \
                'classifier__C': np.logspace(0, 4, 10)}, \
                \
                {'classifier': [RandomForestClassifier()], \
                'classifier__n_estimators': [10, 100, 1000], \
                'classifier__max_featrures': [1, 2, 3]}]

gridsearch = GridSearchCV(pipe, search_space, cv=5, verbose=0)

best_model = gridsearch.fit(features, target)
```

### 12.4 selecting best models when preprocessing

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Create a preprocessing object that includes StandardScaler features and PCA
preprocess = FeatureUnion([("std", StandardScaler()), ("pca", PCA())])

# create a pipeline
pipe = Pipeline([("preprocess", preprocess),
                 ("classifier", LogisticRegression())])

# Create space of candidate values
search_space = [{"preprocess__pca__n_components": [1, 2, 3],
                 "classifier__penalty": ["l1", "l2"],
                 "classifier__C": np.logspace(0, 4, 10)}]

clf = GridSearchCV(pipe, search_space, cv=5, verbose=0)

best_model = clf.fit(features, target)
```

### 12.5 speeding up model selection with parallelization

```python
# Load libraries
from sklearn import linear_model, datasets
# Load data
iris = datasets.load_iris()
features = iris.data
target = iris.target
# Create cross-validated logistic regression
logit = linear_model.LogisticRegressionCV(Cs=100) # Train model, C(s) the penlty strengths of regularization
logit.fit(features, target)
```

### 12.7 evaluating performance after model selection

```python
from sklearn.linear_model import LogisticRegression()
from sklearn.model_selection import GridSearchCV, cross_val_score

logistic = LogisticRegression()

# Create range of 20 candidate values for C
C = np.logspace(0, 4, 20)
# Create hyperparameter options
hyperparameters = dict(C=C)

gridsearch = GridSearchCV(logistic, hyperparameters, cv=5, verbose=0)

cross_val_score(gridsearch, features, target).mean()

best_model = gridsearch.fit(features, target)

scores = cross_val_score(gridsearch, features, target)
```

## 13. Linear regression

### 13.1 fitting a line

```python
from sklearn.linear_model import LinearRegression

model.fit(features, target)
model.intercept_
model.coef_
```

### 13.2 handling interactive effects

```python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

interaction = PolynomialFeatures(degree=3, include_bias=False, interaction_only=True)
features_interaction = interaction.fit_transform(features)

regression = LinearRegression()
regression.fit(features_interaction, target)
```

### 13.2 fitting a nonlinear relationship

```python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

polynomial = PolynomialFeatures(degree=3, include_bias=False)
features_polynomail = polynomial.fit_transform(features)

regression = LinearRegression()
regression.fit(features_polynomial, target)
```

### 13.4 reading variance with regularization

$$
R S S=\sum_{i=1}^{n}\left(y_{i}-\hat{y}_{i}\right)^{2}
$$

$$
\mathrm{RSS}+\alpha \sum_{j=1}^{p} \hat{\beta}_{j}^{2}
$$

$$
\frac{1}{2 n} \mathrm{RSS}+\alpha \sum_{j=1}^{p}\left|\hat{\beta}_{j}\right|
$$

```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sclaer = StandardScaler()
features_standardized = scaler.fit_transform(features)

regression = Ridge(alpha=0.5)
regression.fit(features_standardized, target)
```

```python
from sklearn.linear_model import RidgeCV

redge_cv = RidgeCV(alpha=[0.01, 0.1, 1, 10])
model_cv = redge_cv.fit(features_standardized, target)
model_cv.corf_
model_cv.alpha_
```

### 13.5 reducing features with Lasso regression

```python
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

regression = Lasso(alpha=0.5)
regression.fit(features_standardized, target)

regression.coef_
```

## 15. K-Nearest Neighbors

### 15.1 finding an observation's nearest neighbors

```python
from sklearn.neighbors import NearestNeighbors

nearest_neighbors = NearestNeighbors(n_neighbors=n).fit(features_standardized)

new_observations = [1, 1, 1, 1]
distance, indices = nearest_neighbors.kneighbors([new_observation])
```

$$
d_{\text {eudidean}}=\sqrt{\sum_{i=1}^{n}\left(x_{i}-y_{i}\right)^{2}}
$$

$$
d_{\text {manhattan}}=\sum_{i=1}^{n}\left|x_{i}-y_{i}\right|
$$

$$
d_{\text {minkowski}}=\left(\sum_{i=1}^{n}\left|x_{i}-y_{i}\right|^{p}\right)^{1 / p}
$$

In addition, we can use `kneighbors_graph` to create a matrix indicating each observation's nearest neighbors.

### 15.2 creating K-Nearest neighbor classifier

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=n)
knn.fit(x_std, y)

knn.predict(new_observations)

# or predict probability
knn.pred_proba(new_observations)
```

### 15.3 identifying the best neighborhood size

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV

knn = KNeighborsClassifier(n_neighbors=n)

pipe = Pipeline([('standardizer', standardizer), ('knn', knn)])

search_space = [{'knn__n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}]

clf = GridSearchCV(pipe, search_space, cv=n, verbose=0).fit(features_standardized, target)
```

### 15.3 creating a redius-based nearest neighbor classifier

```python
from sklearn.neighbors import RadiusNeighborsClassifier

rnn = RadiusNeighborsClassifier(radius=0.5).fit(features_standardized, target)

rnn.predict(new_observations)
```