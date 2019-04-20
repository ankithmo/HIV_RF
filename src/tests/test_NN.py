from sklearn.datasets import load_breast_cancer
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

cancer = load_breast_cancer()
#dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])
print(cancer.keys())

X, Y, target_names, desc, feature_names, _ = [cancer[key] for key in cancer.keys()]
print(feature_names)
print(X.shape)
print(target_names)
print(Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))
mlp.fit(X_train, Y_train)

predictions = mlp.predict(X_test)
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))
