import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import dilation, square
from skimage.measure import label, regionprops
from skimage.feature import graycomatrix, graycoprops
import pywt
import joblib  # For saving/loading SVM models

# Load image
img_path = 'Malignant/malig1.jpg'  
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(gray, (200, 200))

# Preprocessing
img_double = cv2.medianBlur(gray, 3) / 255.0

# Thresholding
_, bw = cv2.threshold(img_double, 0.7, 1.0, cv2.THRESH_BINARY)
bw = bw.astype(np.uint8)

# Label and Region Properties
labeled = label(bw)
props = regionprops(labeled)

# Extract largest high-density region
areas = [prop.area for prop in props]
solidities = [prop.solidity for prop in props]
tumor_idx = np.argmax([a if s > 0.5 else 0 for a, s in zip(areas, solidities)])
tumor_mask = labeled == (tumor_idx + 1)
tumor_mask = dilation(tumor_mask, square(3))

# Plot tumor mask and boundaries
plt.figure(figsize=(6, 6))
plt.imshow(gray, cmap='gray')
plt.contour(tumor_mask, colors='r')
plt.title('Detected Tumor Boundary')
plt.axis('off')
plt.show()

# DWT decomposition
coeffs = pywt.wavedec2(gray, 'db4', level=3)
features_dwt = []

# Flatten approximation coefficients
features_dwt.append(coeffs[0].flatten())

# Flatten detail coefficients (tuples of LH, HL, HH)
for detail in coeffs[1:]:
    for arr in detail:
        features_dwt.append(arr.flatten())

features_dwt = np.hstack(features_dwt)


G = gray  # reshape roughly to image-like format if needed

# Compute GLCM (Gray-Level Co-occurrence Matrix)
g = graycomatrix(G.astype(np.uint8), distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
contrast =  graycoprops(g, 'contrast')[0, 0]
correlation =  graycoprops(g, 'correlation')[0, 0]
energy =  graycoprops(g, 'energy')[0, 0]
homogeneity = graycoprops(g, 'homogeneity')[0, 0]

# Statistical Features
mean = np.mean(G)
std_dev = np.std(G)
entropy_val = -np.sum(G * np.log2(G + 1e-10))
rms = np.sqrt(np.mean(G ** 2))
variance = np.var(G)
smoothness = 1 - (1 / (1 + np.sum(G)))
kurtosis_val = np.mean((G - mean) ** 4) / (std_dev ** 4)
skewness_val = np.mean((G - mean) ** 3) / (std_dev ** 3)

# Inverse Difference Moment
idm = np.sum(G / (1 + (np.arange(G.shape[1]) - np.arange(G.shape[1]).reshape(-1, 1)) ** 2))

# Final feature vector
feature_vector = np.array([contrast, correlation, energy, homogeneity,
                           mean, std_dev, entropy_val, rms, variance,
                           smoothness, kurtosis_val, skewness_val, idm]).reshape(1, -1)



# Load scaler and model
scaler = joblib.load("scaler.pkl")
model = joblib.load("svm_model.pkl")

# Scale and predict
feature_vector_scaled = scaler.transform(feature_vector)
prediction = model.predict(feature_vector_scaled)

print("Tumor Type:", "MALIGNANT" if prediction[0] == 1 else "BENIGN")
