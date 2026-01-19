import os
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

# Extract 13 features: GLCM + statistical texture metrics
def extract_features(gray):
    g = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(g, 'contrast')[0, 0]
    correlation = graycoprops(g, 'correlation')[0, 0]
    energy = graycoprops(g, 'energy')[0, 0]
    homogeneity = graycoprops(g, 'homogeneity')[0, 0]

    mean = np.mean(gray)
    std_dev = np.std(gray)
    entropy_val = -np.sum(gray * np.log2(gray + 1e-10))
    rms = np.sqrt(np.mean(gray ** 2))
    rms =np.cube(np.mean(gray**))
    variance = np.var(gray)
    smoothness = 1 - (1 / (1 + np.sum(gray)))
    kurtosis_val = np.mean((gray - mean) ** 4) / (std_dev ** 4)
    skewness_val = np.mean((gray - mean) ** 3) / (std_dev ** 3)
    idm = np.sum(gray / (1 + (np.arange(gray.shape[1]) - np.arange(gray.shape[1]).reshape(-1, 1)) ** 2))

    return [contrast, correlation, energy, homogeneity, mean, std_dev, entropy_val, rms,
            variance, smoothness, kurtosis_val, skewness_val, idm]

# Initialize lists
X = []
y = []

# Folder to label mapping
folders = {
    "benign": 0,
    "malignant": 1
}

# Loop through folders and images
for folder_name, label in folders.items():
    folder_path = os.path.join(folder_name)
    for filename in os.listdir(folder_path):
        if filename.endswith((".jpg", ".png", ".bmp")):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (200, 200))
                try:
                    features = extract_features(gray)
                    X.append(features)
                    y.append(label)
                except Exception as e:
                    print(f" Skipped {img_path}: {e}")

# Convert to arrays and save
X = np.array(X)
y = np.array(y)

np.save("X_features.npy", X)
np.save("y_labels.npy", y)
print(f" Dataset created and saved! Shape: {X.shape}, Labels: {np.bincount(y)}")