import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data = []
labels = []

dataset_path = "dataset"

categories = ["pizza", "burger"]

# Load images
for category in categories:

    path = os.path.join(dataset_path, category)

    label = categories.index(category)

    for image_name in os.listdir(path):

        image_path = os.path.join(path, image_name)

        image = cv2.imread(image_path)

        image = cv2.resize(image, (64, 64))

        data.append(image.flatten())

        labels.append(label)

# Convert into arrays
X = np.array(data)
y = np.array(labels)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = SVC()

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)