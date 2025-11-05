mport tensorflow as tf
import os

def load_data(train_dir, img_size=(128, 128), batch_size=32, validation_split=0.2, seed=42):
    # Rescale layer
    rescale = tf.keras.layers.Rescaling(1./255)

    # Data Augmentation layers
    # Note: ImageDataGenerator's shear and fill_mode='nearest' are harder to replicate exactly with simple layers.
    # For 'fill_mode', the random translation layers often handle this implicitly by extending borders or filling with black.
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomTranslation(height_factor=0.2, width_factor=0.2),
    ])

    # Load training data
    train_data = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='binary',
        image_size=img_size,
        interpolation='nearest',
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        validation_split=validation_split,
        subset='training'
    )

    # Load validation data
    val_data = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='binary',
        image_size=img_size,
        interpolation='nearest',
        batch_size=batch_size,
        shuffle=False, # No need to shuffle validation data
        seed=seed,
        validation_split=validation_split,
        subset='validation'
    )

    # Apply rescaling and augmentation
    # Map the rescaling to both training and validation datasets
    train_data = train_data.map(lambda x, y: (rescale(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    val_data = val_data.map(lambda x, y: (rescale(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    # Apply data augmentation only to the training dataset
    train_data = train_data.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)


    # Cache and prefetch for performance
    train_data = train_data.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_data = val_data.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_data, val_data
