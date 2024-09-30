import tkinter as tk
from tkinter import filedialog, messagebox
import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf
from scipy.io.wavfile import write

# Load your pre-trained model
model = tf.keras.models.load_model(r"path_of_file\EmotionSpeech.h5")

# Define the emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Function to extract features from audio
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs = np.mean(mfccs.T, axis=0)
    return mfccs

# Function to predict emotion
def predict_emotion(features):
    features = np.expand_dims(features, axis=0)
    prediction = model.predict(features)
    emotion_index = np.argmax(prediction)
    emotion = emotion_labels[emotion_index]
    return emotion

# Function to handle file upload
def upload_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        features = extract_features(file_path)
        emotion = predict_emotion(features)
        messagebox.showinfo("Prediction", f"Predicted Emotion: {emotion}")

# Function to record audio
def record_audio():
    fs = 44100  # Sample rate
    seconds = 5  # Duration of recording
    messagebox.showinfo("Recording", "Recording for 5 seconds...")
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished
    write('output.wav', fs, myrecording)  # Save as WAV file
    features = extract_features('output.wav')
    emotion = predict_emotion(features)
    messagebox.showinfo("Prediction", f"Predicted Emotion: {emotion}")

# Create the main window
root = tk.Tk()
root.title("Emotion Detection through Voice")

# Create buttons
upload_button = tk.Button(root, text="Upload Voice Note", command=upload_file)
upload_button.pack(pady=20)

record_button = tk.Button(root, text="Record Voice", command=record_audio)
record_button.pack(pady=20)

# Run the application
root.mainloop()
