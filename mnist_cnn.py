import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load and preprocess the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

# Evaluate on test data
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc:.4f}')

from PIL import Image
import numpy as np

# Load and preprocess your own image
img = Image.open("my_digit.png").convert("L")  # Convert to grayscale
img = img.resize((28, 28))                     # Resize to 28x28
img_array = np.array(img)

# Invert colors if background is white and digit is dark
img_array = 255 - img_array

# Normalize and reshape
img_array = img_array / 255.0
img_array = img_array.reshape(1, 28, 28, 1)

# Predict using the trained model
prediction = model.predict(img_array)
predicted_digit = np.argmax(prediction)
print("Predicted digit from your image:", predicted_digit)
