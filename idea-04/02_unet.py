
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf


class AnnotationGenerator():
    def __init__(self, root: Path):
        self.root = root

    def _create_dir(self, path):
        file_path = self.root / path
        file_path.mkdir(parents=True, exist_ok=True)

    def _create_file(self, path):
        file_path = self.root / path
        file_dir = file_path.parent
        file_dir.mkdir(parents=True, exist_ok=True)
        file_path.touch()
        return file_path
    
    @staticmethod
    def _format_data(frames):
        # The data coming in seems to be in the range 0.0 to 1.0
        # We convert it to 0..255 unsigned 8-bit integers so will
        # be saved as a black/white image.
        min_ = frames.min()
        max_ = frames.max()
        return ((frames - min_) * (1.0/(max_ - min_) * 255.0)).astype('uint8')

    @staticmethod
    def _draw_defects_as_cross(frame, defects):
        rgb = np.zeros([frame.shape[0],frame.shape[1],3])
        rgb[:,:,1] = frame.astype('uint8')
        rgb[:,:,2] = frame.astype('uint8')
        rgb[:,:,0] = frame.astype('uint8')
        marker_color = (0,0,255)
        line_thickness = 1
        marker_size = 2
        for polygon in defects:
            for x, y in polygon:
                cv2.line(rgb, (int(y-marker_size),int(x)), (int(y+marker_size),int(x)), marker_color, line_thickness)
                cv2.line(rgb, (int(y),int(x-marker_size)), (int(y),int(x+marker_size)), marker_color, line_thickness)
        return rgb

    @staticmethod
    def _draw_defects_circle(frame, defects, edge_radius, center_radius):
        rgb = np.zeros([frame.shape[0],frame.shape[1],3])
        rgb[:,:,1] = frame.astype('uint8')
        rgb[:,:,2] = frame.astype('uint8')
        rgb[:,:,0] = frame.astype('uint8')
        center_color = (0, 255, 0)
        edge_color = (0, 0, 255)
        line_thickness = 1

        for polygon in defects:
            for x, y in polygon:
                cv2.circle(rgb, (int(y), int(x)), edge_radius, edge_color, thickness=-1)

        for polygon in defects:
            for x, y in polygon:
                cv2.circle(rgb, (int(y), int(x)), center_radius, center_color, thickness=-1)

        return rgb

    @staticmethod
    def _draw_defects_circle_uint8(frame, defects, edge_radius, center_radius):
        rgb = np.zeros([frame.shape[0],frame.shape[1],1])
        center_color = 128
        edge_color = 255
        line_thickness = 1

        for polygon in defects:
            for x, y in polygon:
                cv2.circle(rgb, (int(y), int(x)), edge_radius, edge_color, thickness=-1)

        for polygon in defects:
            for x, y in polygon:
                cv2.circle(rgb, (int(y), int(x)), center_radius, center_color, thickness=-1)

        return rgb

    @staticmethod
    def _fill_nan_with(arr, value):
        mask = np.isnan(arr)
        arr[mask] = value
        return arr

    def _local_enhancements(self, frame):
        original_type = frame.dtype
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
        frame = clahe.apply(self._format_data(frame))
        return frame.astype(original_type) / frame.max()

    def annotate(self, frame, defects):
        img = self._format_data(self._local_enhancements(np.copy(frame)))
        return self._draw_defects_circle(frame, defects, 16, 12)
        # return self._draw_defects_as_cross(img, defects)

    def save(self, name, img):
        fn = self._create_file('%s.png' % name)
        cv2.imwrite(fn.as_posix(), img)

# frames = np.load('../dataset/Graphene_CrSi.npy')
# defects = np.load('../dataset/topo_defects.npy', allow_pickle=True)
# defects = defects[()]
# a = AnnotationGenerator(Path('.'))
# for index, frame in enumerate(frames):
#     img = a.annotate(frame, defects[index])
#     a.save('original/%03d' % index, a._format_data(frame))
#     a.save('annotations/tri/r_16_12/%03d' % index, img)
#     # annotation = generate_annotations(frame, defects[index], 5.0)
#     print('Ran frame %03d' % index)

import random
random.seed(12345)

from collections import namedtuple
Sample = namedtuple("Sample", "image mask")

class SegmentationTrainer:
    def __init__(self, root: Path):
        self.root = root
        self.training_set = None
        self.test_set = None
        self.validation_set = None
        self.samples = []

        self._training_ds = None
        self._testing_ds = None

        self._buffer_size = 1024
        self._seed = 1234
        self._batch_size = 32

    def limit_samples(self, max_count):
        self.samples[:] = self.samples[0:max_count]

    def clear_samples(self):
        self.samples = []

    def find_samples(self, image_path: Path, mask_path: Path, pattern='*'):
        image_dir = self.root / image_path
        mask_dir = self.root / mask_path
        images = sorted(image_dir.glob(pattern))
        masks = sorted(mask_dir.glob(pattern))
        for i, m in zip(images, masks):
            if i.stem != m.stem:
                print('Possible mismatch %s (original) == %s (annotation) ? FALSE' % (i.stem, m.stem))
            self.samples.append(Sample(i, m))

    def split_train_test(self, train_size=0.7, test_size=0.2, validation_size=0.1):
        train_count = int(train_size*len(self.samples))
        test_count = int(test_size*len(self.samples))
        random.shuffle(self.samples)
        self.training_set = self.samples[0:train_count]
        self.test_set = self.samples[train_count:train_count+test_count]
        self.validation_set = self.samples[train_count+test_count:]
        print('Train / Test / Validate Counts: %d / %d / %d' % (
            len(self.training_set),
            len(self.test_set),
            len(self.validation_set),
        ))

    def load_and_prep_sample(self, sample: Sample):
        return self._load_and_prep_image(sample.image), self._load_and_prep_image(sample.mask)

    def _load_and_prep_image(self, path: Path):
        img = cv2.imread(path.as_posix(), cv2.IMREAD_GRAYSCALE)
        img = tf.cast(img, tf.float32) / 255.0
        return img

    def _prepare_data_set(self, set_paths):
        samples = [ self.load_and_prep_sample(sample) for sample in set_paths ]
        return tf.data.Dataset.from_tensor_slices(samples)
        # return (
        #     tf.data.Dataset.from_tensor_slices(set_paths)
        #     .shuffle(buffer_size=self._buffer_size, seed=self._seed)
        #     .map(self.load_and_prep_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        #     .batch(self._batch_size)
        #     .prefetch(tf.data.experimental.AUTOTUNE)
        # )

    def prepare_data_sets(self):
        self._training_ds = self._prepare_data_set(self.training_set)
        self._testing_ds = self._prepare_data_set(self.test_set)


# https://idiotdeveloper.com/unet-implementation-in-tensorflow-using-keras-api/

from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model

def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_unet(input_shape):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024)

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="U-Net")
    return model

import time
start = time.time()
trainer = SegmentationTrainer(Path('.'))
trainer.find_samples(Path('original'), Path('annotations/tri/r_16_12'))
trainer.limit_samples(10)
trainer.split_train_test()
trainer.prepare_data_sets()
end1 = time.time()
print('Duration: %ds' % (end1 - start))

OUTPUT_CHANNEL = 3

tf.keras.backend.clear_session()
input_shape = (1024, 1024, 1)
model = build_unet(input_shape)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
model.summary()

with open('model_summary', 'w') as f:
    model.summary(line_length=200, print_fn=lambda x: f.write('%s\n'%x))
# There is an error:
#   ValueError: No gradients provided for any variable:
# Not sure how to fix it so... I'll try a different model
_ = model.fit(
    trainer._training_ds,
    epochs=3,
    validation_data=trainer._testing_ds,
)
end2 = time.time()
print('Duration: %ds' % (end2 - start))
