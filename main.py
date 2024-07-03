#setting up all libraries for NN training and ML
#resources used https://medium.com/@waleedmousa975/building-a-speech-to-text-with-ai-correction-system-a-step-by-step-tutorial-using-deepspeech-in-a788d6d3129
#resources used https://www.tensorflow.org/tutorials/audio/simple_audio
#resources used https://www.geeksforgeeks.org/audio-recognition-in-tensorflow/
#resources used https://tracyrenee61.medium.com/how-i-used-tensorflows-deep-neural-network-to-make-predictions-on-letters-of-the-alphabet-ba0f135997b8
import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model, load_model  # Import load_model here
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.image import resize
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Define folder structure
data_dir = "/Users/mr.satyajit/Desktop/Arabic-Alphabet-SR/datset"
classes = ["أ", "ب", "ت", "ث", "ج", "ح", "خ", "د", "ذ", "ر", "ز", "س", "ش", "ص", "ض", "ط", "ظ", "ع", "غ", "ف", "ق", "ك", "ل", "م", "ن", "ه", "و", "ي"]

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
                    mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
                    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
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
    labels = to_categorical(labels, num_classes=len(classes))
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Create a neural network model
    input_shape = X_train[0].shape
    input_layer = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    output_layer = Dense(len(classes), activation='softmax')(x)
    model = Model(input_layer, output_layer)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model and save the training history
    history = model.fit(X_train, y_train, epochs=300, batch_size=50, validation_data=(X_test, y_test))

    # Save the model
    model.save('audio_classification_model.h5')
    
    # Load the saved model
    model = load_model('audio_classification_model.h5')

    # Define the target shape for input spectrograms
    target_shape = (128, 128)

    # Function to preprocess and classify an audio file
    def test_audio(file_path, model):
        # Load and preprocess the audio file
        audio_data, sample_rate = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
        mfcc = resize(np.expand_dims(mfcc, axis=-1), target_shape)
        mfcc = tf.reshape(mfcc, (1,) + target_shape + (1,))
        
        # Make predictions
        predictions = model.predict(mfcc)
        
        # Get the class probabilities
        class_probabilities = predictions[0]
        
        # Get the predicted class index
        predicted_class_index = np.argmax(class_probabilities)
        
        return class_probabilities, predicted_class_index

    # Test audio file
    test_audio_file = '/Users/mr.satyajit/Desktop/Arabic-Alphabet-SR/datset/ي/01_03_ي.wav'
    class_probabilities, predicted_class_index = test_audio(test_audio_file, model)

    # Display results for all classes
    for i, class_label in enumerate(classes):
        probability = class_probabilities[i]
        print(f'Class: {class_label}, Probability: {probability:.4f}')

    # Calculate and display the predicted class and accuracy
    predicted_class = classes[predicted_class_index]
    accuracy = class_probabilities[predicted_class_index]
    print(f'The audio is classified as: {predicted_class}')
    print(f'Accuracy: {accuracy:.4f}')

    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.show()

    # Generate predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

   
