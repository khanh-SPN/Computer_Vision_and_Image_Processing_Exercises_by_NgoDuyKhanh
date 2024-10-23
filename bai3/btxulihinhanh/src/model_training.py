import time
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

def train_svm(X_train, y_train):
    svm = SVC()
    start_time = time.time()
    svm.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time
    return svm, training_time

def train_knn(X_train, y_train, n_neighbors=5):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    start_time = time.time()
    knn.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time
    return knn, training_time

def train_decision_tree(X_train, y_train):
    tree = DecisionTreeClassifier()
    start_time = time.time()
    tree.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time
    return tree, training_time
