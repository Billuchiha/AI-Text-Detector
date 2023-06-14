import tkinter as tk
from tkinter import messagebox
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf

# Load the tokenizer and model for DistilBERT
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = TFAutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased')

# Define the GUI window
window = tk.Tk()
window.title("AI Detection Window")

# Create a label
label = tk.Label(window, text="Enter a text:")
label.pack()

# Create an entry field
entry = tk.Entry(window, width=50)
entry.pack()

# Define the AI detection function
def detect_ai():
    text = entry.get()

    # Preprocess the text
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors='tf')

    # Make predictions using the DistilBERT model
    logits = model(inputs)[0]
    probabilities = tf.nn.softmax(logits, axis=1)
    predicted_class = tf.argmax(probabilities, axis=1).numpy()[0]

    # Display the result
    if predicted_class == 1:
        messagebox.showinfo("AI Detection", "The text is AI-generated.")
    else:
        messagebox.showinfo("AI Detection", "The text is not AI-generated.")

# Create a button
button = tk.Button(window, text="Detect AI", command=detect_ai)
button.pack()

# Run the GUI
window.mainloop()