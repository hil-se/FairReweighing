import os
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split


REPO_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = REPO_ROOT / "data" / "scut" / "vgg_face_weights.h5"
IMAGE_SIZE = 224
EPOCHS = 50
IMAGE_BATCH_SIZE = 16
HEAD_BATCH_SIZE = 128
PATIENCE = 5
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 1e-3


class ScutVGGFaceRegressor(BaseEstimator, RegressorMixin):
    uses_raw_images = True

    def __init__(self, seed=None):
        self.seed = seed

    def fit(self, X, y, sample_weight=None):
        tf = _tensorflow()
        if self.seed is not None:
            tf.keras.utils.set_random_seed(self.seed)

        paths = _paths(X)
        y = np.asarray(y, dtype=np.float32)
        weights = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float32)
        train_idx, val_idx = _split(len(paths), self.seed)

        self.encoder_ = build_vgg_face_feature_extractor()
        train_features = self.encoder_.predict(_dataset(paths[train_idx]), verbose=1)
        val_features = self.encoder_.predict(_dataset(paths[val_idx]), verbose=1)

        train_weights = weights[train_idx] if weights is not None else None
        val_data = (val_features, y[val_idx], weights[val_idx]) if weights is not None else (val_features, y[val_idx])

        self.model_ = build_regression_head(train_features.shape[1])
        self.model_.compile(optimizer=_adam(tf), loss="mse", steps_per_execution=20)
        self.model_.fit(
            train_features,
            y[train_idx],
            sample_weight=train_weights,
            validation_data=val_data,
            batch_size=HEAD_BATCH_SIZE,
            epochs=EPOCHS,
            verbose=1,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, min_delta=1e-4, restore_best_weights=True)],
        )
        return self

    def predict(self, X):
        features = self.encoder_.predict(_dataset(_paths(X)), verbose=0)
        return self.model_.predict(features, batch_size=HEAD_BATCH_SIZE, verbose=0).reshape(-1)


def build_vgg_face_single_encoder():
    tf = _tensorflow()
    L = tf.keras.layers
    encoder = build_vgg_face_feature_extractor()
    inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    output = L.Dense(1)(encoder(inputs, training=False))
    return tf.keras.Model(inputs=inputs, outputs=output)


def build_vgg_face_feature_extractor():
    tf = _tensorflow()
    path = Path(os.environ.get("SCUT_VGG_FACE_WEIGHTS", os.environ.get("SCUT_DATA_ROOT", WEIGHTS_PATH)))
    if path.is_dir():
        path = path / "vgg_face_weights.h5"
    if not path.exists():
        raise FileNotFoundError(f"VGG-Face weights not found: {path}")

    L = tf.keras.layers
    base = tf.keras.Sequential([
        L.ZeroPadding2D((1, 1), input_shape=(224, 224, 3)),
        L.Convolution2D(64, (3, 3), activation="relu"),
        L.ZeroPadding2D((1, 1)),
        L.Convolution2D(64, (3, 3), activation="relu"),
        L.MaxPooling2D((2, 2), strides=(2, 2)),
        L.ZeroPadding2D((1, 1)),
        L.Convolution2D(128, (3, 3), activation="relu"),
        L.ZeroPadding2D((1, 1)),
        L.Convolution2D(128, (3, 3), activation="relu"),
        L.MaxPooling2D((2, 2), strides=(2, 2)),
        L.ZeroPadding2D((1, 1)),
        L.Convolution2D(256, (3, 3), activation="relu"),
        L.ZeroPadding2D((1, 1)),
        L.Convolution2D(256, (3, 3), activation="relu"),
        L.ZeroPadding2D((1, 1)),
        L.Convolution2D(256, (3, 3), activation="relu"),
        L.MaxPooling2D((2, 2), strides=(2, 2)),
        L.ZeroPadding2D((1, 1)),
        L.Convolution2D(512, (3, 3), activation="relu"),
        L.ZeroPadding2D((1, 1)),
        L.Convolution2D(512, (3, 3), activation="relu"),
        L.ZeroPadding2D((1, 1)),
        L.Convolution2D(512, (3, 3), activation="relu"),
        L.MaxPooling2D((2, 2), strides=(2, 2)),
        L.ZeroPadding2D((1, 1)),
        L.Convolution2D(512, (3, 3), activation="relu"),
        L.ZeroPadding2D((1, 1)),
        L.Convolution2D(512, (3, 3), activation="relu"),
        L.ZeroPadding2D((1, 1)),
        L.Convolution2D(512, (3, 3), activation="relu"),
        L.MaxPooling2D((2, 2), strides=(2, 2)),
        L.Convolution2D(4096, (7, 7), activation="relu"),
        L.Dropout(0.5),
        L.Convolution2D(4096, (1, 1), activation="relu"),
        L.Dropout(0.5),
        L.Convolution2D(2622, (1, 1)),
        L.Flatten(),
        L.Activation("softmax"),
    ])
    base.load_weights(str(path))
    for layer in base.layers:
        layer.trainable = False

    _ = base(tf.keras.Input(shape=(224, 224, 3)))
    output = L.Flatten()(base.layers[-4].output)
    return tf.keras.Model(inputs=base.inputs, outputs=output)


def build_regression_head(n_features):
    tf = _tensorflow()
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,)),
        tf.keras.layers.Dense(1),
    ])


def _dataset(paths, y=None, weights=None, shuffle=False, seed=None):
    tf = _tensorflow()
    paths = np.asarray(paths, dtype=str)
    if y is None:
        data = tf.data.Dataset.from_tensor_slices(paths).map(_load_image, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        labels = np.asarray(y, dtype=np.float32)
        if weights is None:
            data = tf.data.Dataset.from_tensor_slices((paths, labels)).map(lambda p, label: (_load_image(p), label), num_parallel_calls=tf.data.AUTOTUNE)
        else:
            weights = np.asarray(weights, dtype=np.float32)
            data = tf.data.Dataset.from_tensor_slices((paths, labels, weights)).map(lambda p, label, w: (_load_image(p), label, w), num_parallel_calls=tf.data.AUTOTUNE)
        if shuffle:
            data = data.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    return data.batch(max(1, min(IMAGE_BATCH_SIZE, len(paths)))).prefetch(tf.data.AUTOTUNE)


def _load_image(path):
    tf = _tensorflow()
    image = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    image.set_shape([None, None, 3])
    return tf.image.resize(tf.cast(image, tf.float32) / 255.0, [IMAGE_SIZE, IMAGE_SIZE])


def _paths(X):
    if hasattr(X, "columns"):
        return X["image_path"].astype(str).to_numpy()
    array = np.asarray(X)
    return array.reshape(-1).astype(str)


def _split(n_samples, seed):
    indices = np.arange(n_samples)
    if n_samples < 3:
        return indices, indices
    return train_test_split(indices, test_size=VALIDATION_SPLIT, random_state=seed)


def _adam(tf):
    if hasattr(tf.keras.optimizers, "legacy"):
        return tf.keras.optimizers.Adam(LEARNING_RATE)
    return tf.keras.optimizers.Adam(LEARNING_RATE)


def _tensorflow():
    import tensorflow as tf

    return tf
