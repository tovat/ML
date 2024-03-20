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

# Fitting the final model to the whole dataset
final_model = SVC(C=0.5, gamma=1, kernel='poly')
final_model.fit(X_scaled, y)

#model = joblib.load("C:/Users/tovat/OneDrive/Dokument/EC_utbildning/MachineLearning/final_model.pkl")


# Save the final model to a temporary location
#temp_model_path = "temp_final_model.pkl"
#joblib.dump(final_model, temp_model_path)

# Move the final model to the large file storage area
#os.system("git lfs track 'final_model.pkl'")
#os.system(f"mv {temp_model_path} final_model.pkl")

joblib.dump(final_model, ("final_model.pkl"))