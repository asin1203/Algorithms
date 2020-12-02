# 4. Scikit-learn

Scikit-Learn, also known as sklearn, is Python’s premier general-purpose machine learning library. While you’ll find other packages that do better at certain tasks, Scikit-Learn’s versatility makes it the best starting place for most ML problems .It’s also a fantastic library for beginners because it offers a high-level interface for many tasks (e.g. preprocessing data, cross-validation, etc.).

Scikit-learn provides a range of supervised and unsupervised learning algorithms via a consistent interface in Python.The library is built upon the **SciPy (Scientific Python)** that must be installed before you can use scikit-learn.

Extensions or modules for SciPy care conventionally named [Scikits](http://scikits.appspot.com/scikits). As such, the module provides learning algorithms and is named scikit-learn.

Some popular groups of models provided by scikit-learn include:

- ***Clustering***: for grouping unlabeled data such as KMeans.
- ***Cross Validation***: for estimating the performance of supervised models on unseen data.
- ***Datasets***: for test datasets and for generating datasets with specific properties for investigating model behavior.
- ***Dimensionality Reduction***: for reducing the number of attributes in data for summarization, visualization and feature selection such as Principal component analysis.
- ***Ensemble methods***: for combining the predictions of multiple supervised models.
- ***Feature extraction***: for defining attributes in image and text data.
- ***Feature selection***: for identifying meaningful attributes from which to create supervised models.
- ***Parameter Tuning***: for getting the most out of supervised models.
- ***Manifold Learning***: For summarizing and depicting complex multi-dimensional data.
- ***Supervised Models***: a vast array not limited to generalized linear models, discriminate analysis, naive bayes, lazy methods, neural networks, support vector machines and decision trees.

## Example: Classification and Regression Trees

An example to show you how easy it is to use the library.

In this example, we use the Classification and Regression decision tree algorithm (ahead in this tutorial) to model the Iris flower dataset (This dataset is provided as an example dataset with the library and is loaded. The classifier is fit on the data and then predictions are made on the training data).

Finally, the classification accuracy and a - [confusion matrix](https://machinelearningmastery.com/confusion-matrix-machine-learning/) is printed.

```py
# Sample Decision Tree Classifier
from sklearn import datasets
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
# load the iris datasets
dataset = datasets.load_iris()
# fit a CART model to the data
model = DecisionTreeClassifier()
model.fit(dataset.data, dataset.target)
print(model)
# make predictions
expected = dataset.target
predicted = model.predict(dataset.data)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
```
Running this example produces the following output, showing you the details of the trained model, the skill of the model according to some common metrics and a confusion matrix.

```
DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=None, splitter='best')
             precision    recall  f1-score   support

          0       1.00      1.00      1.00        50
          1       1.00      1.00      1.00        50
          2       1.00      1.00      1.00        50

avg / total       1.00      1.00      1.00       150

[[50  0  0]
 [ 0 50  0]
 [ 0  0 50]]
 ```

 We'll be using scikit-learn further to train various model on different supervised learning algorithms.

## References
- [Scikit-Learn documentation](https://scikit-learn.org/stable/user_guide.html)