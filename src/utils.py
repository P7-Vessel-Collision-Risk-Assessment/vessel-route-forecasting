import tensorflow as tf
import os


@tf.keras.utils.register_keras_serializable()
def dist_euclidean(y_true, y_pred):
    return tf.keras.backend.mean(
        tf.keras.backend.sqrt(
            tf.keras.backend.sum(
                tf.keras.backend.square(y_pred - y_true), axis=-1, keepdims=True
            )
            + 1e-16
        ),
        axis=-1,
    )


def get_model_path() -> str | None:
    return os.environ.get("MODEL_PATH", None)
