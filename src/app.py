from datetime import datetime
import tensorflow as tf
import argparse
from flask import Flask, request, jsonify

from src.utils import dist_euclidean


def create_app(model_path, debug=False):
    app = Flask(__name__)

    starttime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    model = tf.keras.models.load_model(
        model_path, custom_objects={"dist_euclidean": dist_euclidean}
    )

    config = model.get_config()

    @app.route("/")
    def info():
        return jsonify(
            {
                "model": model_path,
                "input_shape": config["layers"][0]["config"]["batch_shape"],
                "gpus_available": tf.config.list_physical_devices("GPU"),
                "server_info": {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "startup_timestamp": starttime,
                    "debug_mode": debug,
                },
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

    return app


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

    app = create_app(model_path, args.debug)

    print(f"Starting server on {args.host}:{args.port}")
    print(f"Model: {model_path}")

    app.run(
        debug=args.debug,
        port=args.port,
        host=args.host,
    )
