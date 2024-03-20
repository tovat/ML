from sklearn.datasets import fetch_openml
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import os

mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)
X, y = mnist["data"], mnist["target"]

# Standardizing the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, ("scaler.pkl"))

# Fitting the final model to the whole dataset and saving it
final_model = SVC(C=0.5, gamma=1, kernel='poly')
final_model.fit(X_scaled, y)

joblib.dump(final_model, ("final_model.pkl"))