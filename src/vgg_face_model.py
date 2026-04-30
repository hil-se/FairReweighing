import os
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split


REPO_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = REPO_ROOT / "data" / "scut" / "vgg_face_weights.h5"
IMAGE_SIZE = 224
EPOCHS = 100
BATCH_SIZE = 2
PATIENCE = 10
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 1e-4


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

        train_data = _dataset(paths[train_idx], y[train_idx], weights[train_idx] if weights is not None else None, True, self.seed)
        val_data = _dataset(paths[val_idx], y[val_idx], weights[val_idx] if weights is not None else None, False, self.seed)

        self.model_ = build_vgg_face_single_encoder()
        self.model_.compile(optimizer=tf.keras.optimizers.SGD(LEARNING_RATE), loss="mse")
        self.model_.fit(
            train_data,
            validation_data=val_data,
            epochs=EPOCHS,
            verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, min_delta=1e-4, restore_best_weights=True)],
        )
        return self

    def predict(self, X):
        return self.model_.predict(_dataset(_paths(X)), verbose=0).reshape(-1)


def build_vgg_face_single_encoder():
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
        layer.trainable = True

    _ = base(tf.keras.Input(shape=(224, 224, 3)))
    output = L.Dense(1)(L.Flatten()(base.layers[-4].output))
    return tf.keras.Model(inputs=base.inputs, outputs=output)


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
    return data.batch(max(1, min(BATCH_SIZE, len(paths)))).prefetch(tf.data.AUTOTUNE)


def _load_image(path):
    tf = _tensorflow()
    image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
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


def _tensorflow():
    import tensorflow as tf

    return tf
