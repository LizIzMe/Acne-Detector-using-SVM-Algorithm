import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle


def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hist_bgr = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist_hsv = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])

    cv2.normalize(hist_bgr, hist_bgr)
    cv2.normalize(hist_hsv, hist_hsv)

    edges = cv2.Canny(gray, 100, 200)
    edge_histogram = cv2.calcHist([edges], [0], None, [2], [0, 256])
    cv2.normalize(edge_histogram, edge_histogram)

    features = np.concatenate([
        hist_bgr.flatten(),
        hist_hsv.flatten(),
        edge_histogram.flatten()
    ])
    return features

def augment_image(image):
    augmented = []
    augmented.append(cv2.flip(image, 1))
    augmented.append(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))
    bright = cv2.convertScaleAbs(image, alpha=1.2, beta=10)
    augmented.append(bright)
    return augmented

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

dataset_path = r'C:\Users\valman a\PycharmProjects\SVM_Acne_Project\.venv\Created Dataset'
categories = ["Level_0", "Level_1", "Level_2"]

X = []
y = []

for category in categories:
    path = os.path.join(dataset_path, category)
    class_num = categories.index(category)
    for img in load_images_from_folder(path):
        features = extract_features(img)
        X.append(features)
        y.append(class_num)
        for aug_img in augment_image(img):
            aug_features = extract_features(aug_img)
            X.append(aug_features)
            y.append(class_num)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf_classifier = RandomForestClassifier(class_weight='balanced', random_state=42)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, verbose=2, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

print("Best Parameters:", grid_search.best_params_)

best_rf_classifier = grid_search.best_estimator_

cv_scores = cross_val_score(best_rf_classifier, X_train_scaled, y_train, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

best_rf_classifier.fit(X_train_scaled, y_train)

y_pred = best_rf_classifier.predict(X_test_scaled)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=categories))

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Rate: {accuracy * 100:.2f}%")

model_filename = 'acne_severity_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(best_rf_classifier, file)
print(f"Model saved as {model_filename}")

scaler_filename = 'acne_severity_scaler.pkl'
with open(scaler_filename, 'wb') as file:
    pickle.dump(scaler, file)
print(f"Scaler saved as {scaler_filename}")

categories_filename = 'acne_severity_categories.pkl'
with open(categories_filename, 'wb') as file:
    pickle.dump(categories, file)
print(f"Categories saved as {categories_filename}")