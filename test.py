import speech_recognition as sr
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.image import resize

# Define the target shape for input spectrograms
target_shape = (128, 128)

# Define class labels
classes = ["أ", "ب", "ت", "ث", "ج", "ح", "خ", "د", "ذ", "ر", "ز", "س", "ش", "ص", "ض", "ط", "ظ", "ع", "غ", "ف", "ق", "ك", "ل", "م", "ن", "ه", "و", "ي"]

# Load the saved model
model = tf.keras.models.load_model('audio_classification_model.h5')

def preprocess_audio(audio_data, sample_rate):
    mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
    mel_spectrogram = tf.reshape(mel_spectrogram, (1,) + target_shape + (1,))
    return mel_spectrogram

def test_audio(audio_data, sample_rate, model):
    mel_spectrogram = preprocess_audio(audio_data, sample_rate)
    predictions = model.predict(mel_spectrogram)
    class_probabilities = predictions[0]
    predicted_class_index = np.argmax(class_probabilities)
    prediction_rate = class_probabilities[predicted_class_index]  # Prediction rate for the predicted class
    return class_probabilities, predicted_class_index, prediction_rate

def capture_audio_from_microphone():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say an Arabic alphabet:")
        audio = recognizer.listen(source)
        print("Recording complete")

        # Convert the audio to numpy array
        audio_data = np.frombuffer(audio.get_raw_data(), np.int16).astype(np.float32)
        sample_rate = 16000  # Default sample rate for Microphone

        return audio_data, sample_rate

# Capture audio from microphone
audio_data, sample_rate = capture_audio_from_microphone()

# Test the captured audio
class_probabilities, predicted_class_index, prediction_rate = test_audio(audio_data, sample_rate, model)

# Display results for all classes
for i, class_label in enumerate(classes):
    probability = class_probabilities[i]
    print(f'Class: {class_label}, Probability: {probability:.4f}')

# Calculate and display the predicted class and prediction rate
predicted_class = classes[predicted_class_index]
accuracy = class_probabilities[predicted_class_index]
print(f'The audio is identified as: {predicted_class}')
print(f'Recognition Rate: {prediction_rate:.4f}')
