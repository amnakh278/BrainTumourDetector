# BrainTumourDetector
Brain Tumor Classification using Image Processing and Machine Learning

This project focuses on the classification of brain tumors as benign or malignant using digital image processing and machine learning techniques. It includes the complete pipeline—from image preprocessing and feature extraction to training and using a Support Vector Machine (SVM) model for classification.

Project Features

Image Dataset Handling

Images are categorized and stored under class-specific folders such as Benign/.

The datasett.py script handles image loading, preprocessing, and feature extraction.

Model Training

modeltraning.py trains a Support Vector Machine (SVM) classifier on the extracted features.

The trained model and feature scaler are saved as svm_model.pkl and scaler.pkl.

Prediction Module

model.py loads the saved model and performs predictions on new brain scan images.

Data Files and Outputs

X_features.npy and y_labels.npy store the feature vectors and labels for reproducibility.

Pickle files (.pkl) store trained model and preprocessing objects for efficient deployment.

Technologies Used

Python

NumPy

scikit-learn

Image processing libraries (e.g., OpenCV or PIL)

How to Use

Load and preprocess the image dataset using datasett.py.

Train the classification model by running modeltraning.py.

Use model.py to classify new brain scan images using the trained SVM model.

Educational Value

Demonstrates practical application of image processing for medical image analysis.

Implements key machine learning workflows including data preparation, training, evaluation, and model persistence.

Suitable for academic projects in the fields of Digital Image Processing, Machine Learning, or Biomedical Engineering.
