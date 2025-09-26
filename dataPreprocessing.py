#Oil Spill Segmentation - Data Preprocessing & Augmentation

# --- Mount Google Drive ---
"""

from google.colab import drive
drive.mount('/content/drive')


"""# Imports & Config"""

import tensorflow as tf
import os
from glob import glob
from tensorflow.keras.utils import img_to_array, load_img
import numpy as np
import cv2

# Config
IMG_SIZE = (256, 256)
BATCH_SIZE = 8
AUTOTUNE = tf.data.AUTOTUNE

# Paths
base_dir = "/content/drive/MyDrive/oil_spill_dataset"
train_img_dir  = os.path.join(base_dir, "train/images")
train_mask_dir = os.path.join(base_dir, "train/masks")
val_img_dir    = os.path.join(base_dir, "val/images")
val_mask_dir   = os.path.join(base_dir, "val/masks")

"""# Helper Functions"""

def augment_image_mask(image, mask):
    """ Apply random augmentations to image and mask.
    1.horizontal flip
    2.vertical flip
    3.90° rotation
    4.zoom (central crop + resize)
    5.saturation
    6.contrast
    7.brightness """

    # Ensure dtype consistency
    image = tf.cast(image, tf.float32)
    mask  = tf.cast(mask, tf.float32)

    # Random horizontal flip
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask  = tf.image.flip_left_right(mask)

    # Random vertical flip
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask  = tf.image.flip_up_down(mask)

    # Random 90° rotation
    k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k)
    mask  = tf.image.rot90(mask, k)

    # Random zoom (central crop + resize)
    if tf.random.uniform(()) > 0.5:
        scale = tf.random.uniform([], 0.8, 1.0)
        crop_size = tf.cast(scale * tf.cast(tf.shape(image)[0], tf.float32), tf.int32)

        image = tf.image.resize_with_crop_or_pad(image, crop_size, crop_size)
        mask  = tf.image.resize_with_crop_or_pad(mask, crop_size, crop_size)

        image = tf.image.resize(image, IMG_SIZE)
        mask  = tf.image.resize(mask, IMG_SIZE, method="nearest")

    # Image-only color jitter
    image = tf.image.random_brightness(image, 0.1)
    image = tf.image.random_contrast(image, 0.9, 1.1)
    image = tf.image.random_saturation(image, 0.9, 1.1)

    return image, mask


def median_filter(image, ksize=3):
    # image: [H, W, C], float32 [0,1]
    image = tf.expand_dims(image, axis=0)  # [1, H, W, C]

    patches = tf.image.extract_patches(
        images=image,
        sizes=[1, ksize, ksize, 1],
        strides=[1, 1, 1, 1],
        rates=[1, 1, 1, 1],
        padding="SAME"
    )  # [1, H, W, ksize*ksize*C]

    H = tf.shape(image)[1]
    W = tf.shape(image)[2]
    C = tf.shape(image)[3]

    # Reshape so each channel is separate: [H, W, C, ksize*ksize]
    patches = tf.reshape(patches, [1, H, W, ksize*ksize, C])

    # Take median along window dimension
    patches = tf.sort(patches, axis=3)  # sort along patch axis
    med = patches[:, :, :, ksize*ksize//2, :]  # pick median

    med_img = tf.squeeze(med, axis=0)  # [H, W, C]

    return med_img

def normalize_image_mask(image, mask):
    """Resize and normalize image; resize mask with nearest neighbor."""
    image = tf.cast(image, tf.float32)
    mask  = tf.cast(mask, tf.float32)

    # Resize
    image = tf.image.resize(image, IMG_SIZE)
    mask  = tf.image.resize(mask, IMG_SIZE, method="nearest")

    # Normalize image
    image = image / 255.0

    # Apply median filter
    #image = median_filter(image, ksize=3)

    # Binarize mask
    mask = mask / 255.0
    mask = tf.where(mask > 0.5, 1.0, 0.0)

    return image, mask

def load_and_preprocess(img_path, mask_path, training=False):
    """Load, decode, and preprocess a single image-mask pair."""
    # Read image
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)

    # Read mask
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)

    # Augment only during training
    if training:
        img, mask = augment_image_mask(img, mask)

    img, mask = normalize_image_mask(img, mask)
    return img, mask

# for visualization
def unnormalize(img):
    img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    img = tf.clip_by_value(img, 0.0, 1.0)   # TF equivalent of np.clip
    return img

"""# Build Dataset"""

# Collect file paths
train_img_paths = sorted(glob(os.path.join(train_img_dir, "*.jpg")))
train_mask_paths = sorted(glob(os.path.join(train_mask_dir, "*.png")))
val_img_paths   = sorted(glob(os.path.join(val_img_dir, "*.jpg")))
val_mask_paths  = sorted(glob(os.path.join(val_mask_dir, "*.png")))

print("Train images:", len(train_img_paths))
print("Train masks :", len(train_mask_paths))
print("Val images  :", len(val_img_paths))
print("Val masks   :", len(val_mask_paths))

# Build tf.data pipelines
train_ds = tf.data.Dataset.from_tensor_slices((train_img_paths, train_mask_paths))
val_ds   = tf.data.Dataset.from_tensor_slices((val_img_paths, val_mask_paths))

train_ds = (train_ds
            .map(lambda x,y: load_and_preprocess(x,y,True), num_parallel_calls=AUTOTUNE)
            .batch(BATCH_SIZE)
            .prefetch(AUTOTUNE))

val_ds = (val_ds
          .map(lambda x,y: load_and_preprocess(x,y,False), num_parallel_calls=AUTOTUNE)
          .batch(BATCH_SIZE)
          .prefetch(AUTOTUNE))

"""# Visualization"""

import matplotlib.pyplot as plt
def visualize_batch(dataset, n=3):
    """Show sample batch of images and masks."""
    for images, masks in dataset.take(1):
        plt.figure(figsize=(12,6))
        for i in range(n):
            plt.subplot(2,n,i+1)
            plt.imshow(unnormalize(images[i]).numpy())
            plt.title("Image")
            plt.axis("off")

            plt.subplot(2,n,n+i+1)
            plt.imshow(masks[i].numpy().squeeze(), cmap="gray")
            plt.title("Mask")
            plt.axis("off")
        plt.show()

def visualize_comparison(img_path, mask_path):
    """Compare raw vs augmented samples."""
    raw_img = tf.image.decode_jpeg(tf.io.read_file(img_path), channels=3)
    raw_mask = tf.image.decode_png(tf.io.read_file(mask_path), channels=1)

    aug_img, aug_mask = augment_image_mask(raw_img, raw_mask)
    aug_img, aug_mask = normalize_image_mask(aug_img, aug_mask)

    plt.figure(figsize=(8,4))
    plt.subplot(2,2,1); plt.imshow(raw_img.numpy().astype("uint8")); plt.title("Raw Image"); plt.axis("off")
    plt.subplot(2,2,2); plt.imshow(raw_mask.numpy().squeeze(), cmap="gray"); plt.title("Raw Mask"); plt.axis("off")
    plt.subplot(2,2,3); plt.imshow(unnormalize(aug_img).numpy()); plt.title("Aug Image"); plt.axis("off")
    plt.subplot(2,2,4); plt.imshow(aug_mask.numpy().squeeze(), cmap="gray"); plt.title("Aug Mask"); plt.axis("off")
    plt.show()

visualize_batch(train_ds, n=4)
visualize_comparison(train_img_paths[0], train_mask_paths[0])
visualize_comparison(train_img_paths[10], train_mask_paths[10])
visualize_comparison(train_img_paths[109], train_mask_paths[109])
visualize_comparison(train_img_paths[3], train_mask_paths[3])
visualize_comparison(train_img_paths[697], train_mask_paths[697])

for img, mask in train_ds.take(1):
    print("Image mean:", tf.reduce_mean(img).numpy())
    print("Image std:", tf.math.reduce_std(img).numpy())