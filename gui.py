# %%
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from datetime import datetime, time
import tensorflow as tf

# %%
# Load the saved model
asl_image_model = tf.keras.models.load_model('sign_language.model.h5')

# %%
# Define the class indices (must match the training setup)
class_indices = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25, 'del': 26, 'nothing': 27, 'space': 28}
class_labels = {v: k for k, v in class_indices.items()}

# %%
def is_within_time_range(start_time, end_time):
    current_time = datetime.now().time()
    return start_time <= current_time <= end_time

# %%
def load_and_preprocess_image(img_path):
    img = Image.open(img_path)
    img = img.resize((64, 64))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# %%
def predict_image(image_path, model):
    if not is_within_time_range(time(18, 0), time(22, 0)):
        return "Predictions are only available between 6 PM and 10 PM"
    
    img = load_and_preprocess_image(image_path)
    prediction = model.predict(img)
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = class_labels[predicted_class_index]
    return predicted_class_label

# %%
class ASLPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.geometry('800x600')
        self.root.title("ASL Predictor")
        self.image_model = asl_image_model

        self.upload_image_button = tk.Button(root, text="Upload Image", command=self.upload_image)
        self.upload_image_button.pack()

        self.result_label = tk.Label(root, text="")
        self.result_label.pack()

        self.image_label = tk.Label(root)
        self.image_label.pack()

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.display_image(file_path)
            prediction = predict_image(file_path, self.image_model)
            self.result_label.config(text=f"Prediction: {prediction}")

    def display_image(self, file_path):
        img = Image.open(file_path)
        img = img.resize((250, 250))
        img = ImageTk.PhotoImage(img)
        self.image_label.config(image=img)
        self.image_label.image = img  # Keep a reference to avoid garbage collection

root = tk.Tk()
app = ASLPredictorApp(root)
root.mainloop()


