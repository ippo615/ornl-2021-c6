import pathlib
import os

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


from tensorflow.python.keras.backend import relu

class Something():
    def __init__(self, image_dir):
        self.data_dir = pathlib.Path(image_dir)
        self.batch_size = 128
        self.img_height = 64
        self.img_width = 64
        self.seed = 123
        self.validation_split = 0.2
        self.model = None
        self.history = None

    def train(self, epochs=10):
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            validation_split=self.validation_split,
            subset="training",
            seed=self.seed,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            label_mode='binary'
        )
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            validation_split=self.validation_split,
            subset="validation",
            seed=self.seed,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            label_mode='binary'
        )

        # Autotune
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        # The nueral network model
        data_augmentation = keras.Sequential([
            layers.experimental.preprocessing.RandomFlip(
                "horizontal", 
                input_shape=(self.img_height, self.img_width, 3)
            ),
            layers.experimental.preprocessing.RandomRotation(0.1),
            layers.experimental.preprocessing.RandomZoom(0.1),
        ])
        self.model = Sequential([
            # data_augmentation,
            layers.experimental.preprocessing.Rescaling(1./255),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])

        # Also note:
        # https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
        self.model.compile(
            optimizer='adam',
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=[
                tf.keras.metrics.Precision(),
                tf.keras.metrics.FalseNegatives(),
            ]
        )

        # Train the model
        epochs=epochs
        self.history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs
        )

    def test_image(self, filename):
        img = keras.preprocessing.image.load_img(filename)
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch
        predictions = self.model.predict(img_array)
        return predictions
        return (predictions > 0.5).astype("int32") 
        # print(model.predict_classes(img_array))
        # print((predictions > 0.5).astype("int32"))
        # print(predictions)

    def model_save(self, filename):
        self.model.save(filename)

    def model_load(self, filename):
        self.model = keras.models.load_model(filename)


# s = Something('08_fourier/')
# s.model_load('08_fourier_model')
# # s.train(epochs=20)
# print(s.test_image('08_fourier/defect/000_0384_0256_full.png'))
# print(s.test_image('08_fourier/normal/000_0128_0192_full.png'))
# # s.model_save('08_fourier_model')

# import glob
# defects = [s.test_image(f)[0][0] for f in list(glob.glob('08_fourier/defect/*.png'))[500:700]]
# normals = [s.test_image(f)[0][0] for f in list(glob.glob('08_fourier/normal/*.png'))[500:700]]

s = Something('07_corrected/')
s.model_load('07_corrected_model')
#s.train(epochs=20)
#print(s.test_image('08_fourier/defect/000_0384_0256_full.png'))
#print(s.test_image('08_fourier/normal/000_0128_0192_full.png'))
#s.model_save('07_corrected_model')

import glob
defects = [s.test_image(f)[0][0] for f in list(glob.glob('07_corrected/defect/*.png'))[000:6000]]
normals = [s.test_image(f)[0][0] for f in list(glob.glob('07_corrected/normal/*.png'))[000:6000]]

from matplotlib import pyplot
bins = np.linspace(0.0, 1.0, 100)

pyplot.hist(defects, bins, alpha=0.5, label='defects')
pyplot.hist(normals, bins, alpha=0.5, label='normals')
# plt.hist([defects, normals], bins, label=['defects', 'normals'])
pyplot.legend(loc='upper right')
pyplot.show()

exit(0)

# data_dir = pathlib.Path('02_split_grid/')
# data_dir = pathlib.Path('05_split_fourier/')
# data_dir = pathlib.Path('06_split_centered/')
# data_dir = pathlib.Path('07_corrected/')
data_dir = pathlib.Path('08_fourier/')

batch_size = 128
img_height = 64
img_width = 64

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='binary'
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='binary'
)

# Just to make sure we've loaded stuff correctly
class_names = train_ds.class_names
print(class_names)

# Also to make sure we've loaded stuff correctly
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(class_names[labels[i]])
#         plt.axis("off")

# Also to make sure we've loaded stuff correctly
for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

# Autotune
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Put things in range 0-1 (from 0-255)
# normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

# normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
# image_batch, labels_batch = next(iter(normalized_ds))
# Notice the pixels values are now in `[0,1]`.
# first_image = image_batch[0]
# print(np.min(first_image), np.max(first_image))

# The nueral network model
num_classes = len(class_names)
data_augmentation = keras.Sequential([
    layers.experimental.preprocessing.RandomFlip(
        "horizontal", 
        input_shape=(img_height, img_width, 3)
    ),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
])
model = Sequential([
    # data_augmentation,
    layers.experimental.preprocessing.Rescaling(1./255),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Also note:
# https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[
        tf.keras.metrics.Precision(),
        tf.keras.metrics.FalseNegatives(),
    ]
)

# Train the model
epochs=20
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# Visualize the results:
acc = history.history['precision']
val_acc = history.history['val_precision']

loss = history.history['false_negatives']
val_loss = history.history['val_false_negatives']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Precision')
plt.plot(epochs_range, val_acc, label='Validation Precision')
plt.legend(loc='lower right')
plt.title('Training and Validation Precision')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training False Negatives')
plt.plot(epochs_range, val_loss, label='Validation False Negatives')
plt.legend(loc='upper right')
plt.title('Training and Validation False Negatives')
plt.show()
# --

# https://stackoverflow.com/questions/57519937/how-to-use-model-predict-for-predicting-single-image-in-tf-keras
def test_image(model, filename):
    img = keras.preprocessing.image.load_img(filename)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = model.predict(img_array)
    print(model.predict_classes(img_array))
    print((predictions > 0.5).astype("int32"))
    print(predictions)

test_image(model, '08_fourier/defect/000_0384_0256_full.png')
test_image(model, '08_fourier/normal/000_0128_0192_full.png')

'''
print('WUT?')
image = train_ds.take(1)
print(image)
w,h = image.shape
img = np.reshape(image, (w*h, 1))
print(model.predict(image))
print((model.predict(image)> 0.5).astype("int32"))
'''

'''
sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)

img = keras.preprocessing.image.load_img(
    sunflower_path, target_size=(img_height, img_width)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
'''

# Save and load the model
# model = ...  # Get model (Sequential, Functional Model, or Model subclass)
# model.save('path/to/location')
# from tensorflow import keras
# model = keras.models.load_model('path/to/location')
