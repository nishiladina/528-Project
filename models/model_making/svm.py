"""
import numpy as np
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# Load data
X = np.load("X.npy")
y = np.load("y.npy")

# Model
model = make_pipeline(StandardScaler(), SVC(kernel='rbf'))

# Cross-validation accuracy
scores = cross_val_score(model, X, y, cv=5)
print("Accuracy per fold:", scores)
print("Mean accuracy:", scores.mean())

# Confusion matrix
y_pred = cross_val_predict(model, X, y, cv=5)
cm = confusion_matrix(y, y_pred)

disp = ConfusionMatrixDisplay(cm, display_labels=["left", "right", "up", "down", "right_lean", "left_lean", "clockwise", "counter_clockwise", "no_movement"])
disp.plot()

plt.title("Cross-Validated Confusion Matrix")
plt.show()
plt.show(block=True)
"""
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline 
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

X = np.load("X.npy")
y = np.load("y.npy")

def build_model():
    """
    Soft-voting ensemble of:
      - RBF SVM  (excellent for high-dim, small datasets)
      - Random Forest (captures non-linear feature interactions)
    Both use probability estimates so soft voting is meaningful.
    """
    svm = SVC(
        kernel="rbf", C=10.0, gamma="scale", probability=True, class_weight="balanced"
    )
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
    )
    combined = VotingClassifier(
        estimators=[("svm", svm), ("rf", rf)],
        voting="soft",
    )
    return combined


model = make_pipeline(
    StandardScaler(),
    build_model()
)

model.fit(X, y)

joblib.dump(model, "combined_model.joblib")
print("Saved model to combined_model.joblib")