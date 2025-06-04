import os
import cv2
import glob
import numpy as np
from tqdm import tqdm
import random
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K
# Shadow-aware CLAHE
def apply_shadow_aware_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    if np.mean(l) < 85:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    else:
        return image

def apply_light_region_denoising(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, _, _ = cv2.split(lab)
    return np.mean(l) > 180
# Combined loss function: Perceptual + SSIM + MSE
class CombinedLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.4, beta=0.3, gamma=0.3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.vgg = self._build_vgg()

    def _build_vgg(self):
        vgg = VGG19(include_top=False, weights='imagenet', input_shape=(None, None, 3))
        vgg.trainable = False
        model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
        return model

    def perceptual_loss(self, y_true, y_pred):
        y_true_vgg = preprocess_input(y_true * 255.0)
        y_pred_vgg = preprocess_input(y_pred * 255.0)
        true_features = self.vgg(y_true_vgg)
        pred_features = self.vgg(y_pred_vgg)
        return tf.reduce_mean(tf.square(true_features - pred_features))

    def ssim_loss(self, y_true, y_pred):
        return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

    def mse_loss(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))

    def call(self, y_true, y_pred):
        pl = self.perceptual_loss(y_true, y_pred)
        sl = self.ssim_loss(y_true, y_pred)
        ml = self.mse_loss(y_true, y_pred)
        return self.alpha * pl + self.beta * sl + self.gamma * ml

# DnCNN model for RGB images
def dncnn_model_rgb(depth=17, filters=64, image_channels=3, kernel_size=3):
    input_img = Input(shape=(None, None, image_channels))
    x = Conv2D(filters, kernel_size, padding='same', activation='relu')(input_img)

    for _ in range(depth - 2):
        x = Conv2D(filters, kernel_size, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    x = Conv2D(image_channels, kernel_size, padding='same')(x)
    output = Add()([input_img, x])
    return Model(inputs=input_img, outputs=output)

# Mixed noise function for RGB images
def add_mixed_noise_rgb(image, salt_prob=0.01, pepper_prob=0.01, gauss_mean=0, gauss_std=10):
    noisy = np.copy(image)
    h, w, c = image.shape
    total = h * w
    num_salt = int(total * salt_prob)
    num_pepper = int(total * pepper_prob)

    for i in range(c):
        salt_coords = [np.random.randint(0, h, num_salt), np.random.randint(0, w, num_salt)]
        noisy[salt_coords[0], salt_coords[1], i] = 255
        pepper_coords = [np.random.randint(0, h, num_pepper), np.random.randint(0, w, num_pepper)]
        noisy[pepper_coords[0], pepper_coords[1], i] = 0

    gauss = np.random.normal(gauss_mean, gauss_std, image.shape).astype(np.float32)
    noisy = np.clip(noisy + gauss, 0, 255).astype(np.uint8)
    return noisy

# MixedNoiseDataGenerator class
class MixedNoiseDataGenerator(Sequence):
    def __init__(self, image_paths, batch_size=8, target_size=(128, 128),
                 salt_prob_range=(0.01, 0.03), pepper_prob_range=(0.01, 0.03),
                 gauss_std_range=(10, 30), shuffle=True):
        self.image_paths = image_paths
        self.batch_size = batch_size
        self.target_size = target_size
        self.salt_prob_range = salt_prob_range
        self.pepper_prob_range = pepper_prob_range
        self.gauss_std_range = gauss_std_range
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.image_paths)

    def __getitem__(self, index):
        batch_paths = self.image_paths[index * self.batch_size:(index + 1) * self.batch_size]
        noisy_images, clean_images = [], []

        for path in batch_paths:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.target_size)
            clean = img.astype(np.float32) / 255.0
            salt = np.random.uniform(*self.salt_prob_range)
            pepper = np.random.uniform(*self.pepper_prob_range)
            std = np.random.uniform(*self.gauss_std_range)
            noisy = add_mixed_noise_rgb(img, salt_prob=salt, pepper_prob=pepper, gauss_std=std).astype(np.float32) / 255.0

            noisy_images.append(noisy)
            clean_images.append(clean)

        return np.array(noisy_images), np.array(clean_images)

# Compile and train the model
model = dncnn_model_rgb()
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=CombinedLoss())

# Update the dataset folder path to your clean images
dataset_folder = r'C:\Users\Students\Desktop\Project\WHU-RS19 -main\WHU-RS19\new_dataset_WHU_RS'  # dataset
image_paths = glob.glob(os.path.join(dataset_folder, "*.png"))
print("Loaded {} image paths.".format(len(image_paths)))

# Split the data into training and validation sets
train_paths, val_paths = train_test_split(image_paths, test_size=0.2, random_state=42)

# Create data generators that sample a range of noise levels
train_gen = MixedNoiseDataGenerator(train_paths, batch_size=8, target_size=(128,128),
                                    salt_prob_range=(0.1, 0.3),
                                    pepper_prob_range=(0.1, 0.3),
                                    gauss_std_range=(10, 30), shuffle=True)
val_gen = MixedNoiseDataGenerator(val_paths, batch_size=8, target_size=(128,128),
                                  salt_prob_range=(0.1, 0.3),
                                  pepper_prob_range=(0.1, 0.3),
                                  gauss_std_range=(10, 30), shuffle=False)

# Build the model
model = dncnn_model_rgb()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse')
model.summary()

# Define callbacks
checkpoint = ModelCheckpoint("dncnn_pretrained_mixednoise.keras", monitor="val_loss", verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1)
early_stop = EarlyStopping(monitor="val_loss", patience=10, verbose=1)

# Train the model using the generators
history = model.fit(
    train_gen,
    epochs=30,
    validation_data=val_gen,
    callbacks=[checkpoint, reduce_lr, early_stop]
)

