#setting up all libraries for NN training and ML
#resources used https://medium.com/@waleedmousa975/building-a-speech-to-text-with-ai-correction-system-a-step-by-step-tutorial-using-deepspeech-in-a788d6d3129
import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.image import resize
from tensorflow.keras.models import load_model

# Define your folder structure
data_dir = "/workspaces/Arabic-Alphabet-SpeechRecognition/Training/Chapter 1"
classes = ['Chapter 1 Part 1', 'Chapter 1 Part 2', 'Chapter 1 Part 3','Chapter 1 Part 4','Chapter 1 Part 5','Chapter 1 Part 6','Chapter 1 Part 7','Chapter 1 Part 8','Chapter 1 Part 9','Chapter 1 Part 10']

# Load and preprocess audio data
def load_and_preprocess_data(data_dir, classes, target_shape=(128, 128)):
    data = []
    labels = []
    
    for i, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Directory {class_dir} does not exist")
            continue
        
        for filename in os.listdir(class_dir):
            if filename.endswith('.wav'):
                file_path = os.path.join(class_dir, filename)
                print(f"Processing file: {file_path}")
                
                try:
                    audio_data, sample_rate = librosa.load(file_path, sr=None)
                    # Perform preprocessing (e.g., convert to Mel spectrogram and resize)
                    mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
                    mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
                    data.append(mel_spectrogram)
                    labels.append(i)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    
    return np.array(data), np.array(labels)

# Split data into training and testing sets
data, labels = load_and_preprocess_data(data_dir, classes)

if data.size == 0 or labels.size == 0:
    print("No data was loaded. Please check your dataset.")
else:
    labels = to_categorical(labels, num_classes=len(classes))  # Convert labels to one-hot encoding
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Create a neural network model
    input_shape = X_train[0].shape
    input_layer = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    output_layer = Dense(len(classes), activation='softmax')(x)
    model = Model(input_layer, output_layer)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test))

