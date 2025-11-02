# Import Libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np

# Data Preprocessing
train_gen = ImageDataGenerator(rescale=1./255)
train_data = train_gen.flow_from_directory('dataset/train', target_size=(128,128), batch_size=32, class_mode='categorical')

# CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(4, activation='softmax')  # 4 categories: 3 diseases + healthy
])

# Compile and Train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, epochs=10)

# Prediction
test_img = load_img('test_leaf.jpg', target_size=(128,128))
img_array = img_to_array(test_img)
img_array = np.expand_dims(img_array, axis=0) / 255.0
result = model.predict(img_array)
predicted_class = np.argmax(result)
print("Predicted Class:", predicted_class)