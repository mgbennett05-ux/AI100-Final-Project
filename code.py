from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# load data
X, y = load_iris(return_X_y=True)

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# create model
model = LogisticRegression(max_iter=200)

# train model
model.fit(X_train, y_train)

# test accuracy
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)
