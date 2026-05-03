from sklearn.ensemble import ExtraTreesClassifier
import joblib
import numpy as np

X = np.load("X.npy")
y = np.load("y.npy")

model = ExtraTreesClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

model.fit(X, y)

joblib.dump(model, "extra_trees.joblib")
print("Saved extra_trees_model.joblib")