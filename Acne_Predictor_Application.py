import cv2
import numpy as np
import pickle
import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk

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

def load_model():
    with open('acne_severity_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('acne_severity_scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    with open('acne_severity_categories.pkl', 'rb') as file:
        categories = pickle.load(file)
    return model, scaler, categories

def predict_severity(image_path):
    model, scaler, categories = load_model()
    img = cv2.imread(image_path)
    if img is None:
        return "Error: Unable to load image"
    features = extract_features(img)
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    return categories[prediction[0]]

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

root = ctk.CTk()
root.title("ACNE SEVERITY DETECTION APPLICATION")
root.geometry("400x550")

selected_image_path = None

def upload_image():
    global selected_image_path
    selected_image_path = filedialog.askopenfilename()
    if selected_image_path:
        # Display the selected image
        img = Image.open(selected_image_path)
        img = img.resize((300, 300))
        photo = ImageTk.PhotoImage(img)
        image_label.configure(image=photo)
        image_label.image = photo
        result_entry.configure(state="normal")
        result_entry.delete(0, tk.END)
        result_entry.configure(state="readonly")

def classify_image():
    if selected_image_path:
        severity = predict_severity(selected_image_path)
        print(f"Predicted Severity: {severity}")
        result_entry.configure(state="normal")
        result_entry.delete(0, tk.END)
        result_entry.insert(0, f"Detected Severity: {severity}")
        result_entry.configure(state="readonly")
    else:
        result_entry.configure(state="normal")
        result_entry.delete(0, tk.END)
        result_entry.insert(0, "Please upload an image first")
        result_entry.configure(state="readonly")

title_label = ctk.CTkLabel(root, text="ACNE SEVERITY DETECTION\nAPPLICATION", font=("Arial", 20, "bold"))
title_label.pack(pady=20)

image_frame = ctk.CTkFrame(root, width=300, height=300)
image_frame.pack(pady=10)

image_label = tk.Label(image_frame, bg="white")
image_label.place(relx=0.5, rely=0.5, anchor="center")


result_entry = ctk.CTkEntry(root, width=300, height=30, state="readonly", justify="center")
result_entry.pack(pady=10)

button_frame = ctk.CTkFrame(root)
button_frame.pack(pady=20, fill="x", padx=20)

upload_button = ctk.CTkButton(button_frame, text="UPLOAD IMAGE", command=upload_image)
upload_button.pack(side="left", expand=True, padx=5)

classify_button = ctk.CTkButton(button_frame, text="CLASSIFY ACNE LEVEL", command=classify_image)
classify_button.pack(side="right", expand=True, padx=5)

root.mainloop()
