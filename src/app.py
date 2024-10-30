import tensorflow as tf
import argparse

# import json
from flask import Flask, request, jsonify

app = Flask(__name__)


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


@app.route("/")
def model_info():
    return jsonify(
        {
            "model": model_path,
            "input_shape": config["layers"][0]["config"]["batch_shape"],
            "gpus_available": tf.config.list_physical_devices("GPU"),
            "test": "test2",
        }
    )


@app.route("/predict", methods=["POST"])
def predict():

    if not request.json:
        return jsonify({"error": "No data provided"})

    data: dict = request.json

    inputs = tf.constant(data.get("data"), dtype=tf.float32)

    pred = model.predict(inputs)

    return jsonify({"prediction": pred.tolist()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", "-m", type=str, required=False)
    parser.add_argument("--debug", action="store_true", required=False)
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    model_path = "rd9_epoch100_h1n350_ffn150_final_model.keras"
    if args.model_path:
        model_path = args.model_path

    model = tf.keras.models.load_model(
        model_path, custom_objects={"dist_euclidean": dist_euclidean}
    )

    config = model.get_config()

    app.run(
        debug=args.debug,
        port=args.port,
        host=args.host,
    )
