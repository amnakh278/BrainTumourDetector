
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
X = np.load("X_features.npy")
y = np.load("y_labels.npy")

# Optional: print class balance 
#bincount count the number of occurences of each value in array
print("Samples per class:", np.bincount(y))

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

rn=RandomForestClassifier()
# Train SVM
model = rn(kernel='linear', probability=True, class_weight='balanced')
model.fit(X_scaled, y)

# Save
joblib.dump(model, "svm_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print(" SVM model and scaler saved.")
