from datetime import datetime
from io import StringIO
import pandas as pd
import tensorflow as tf
import argparse
from flask import Flask, request, jsonify

from utils import dist_euclidean, get_model_path, get_norm_params_path, post_process_prediction, pre_process_data

def create_app(model_path: str, norm_params_path: str, debug=False) -> Flask:
    app = Flask(__name__)

    starttime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    model = tf.keras.models.load_model(  # type: ignore
        model_path, custom_objects={"dist_euclidean": dist_euclidean}
    )

    config = model.get_config()

    norm_params = pd.read_json(norm_params_path)
    norm_params = norm_params.transpose()
    norm_params.columns = ["d", "dlon", "dlat"]

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

        data_json = request.json
        data_df = pd.read_json(StringIO(data_json.get("data")))

        norm_input, data_df = pre_process_data(data_df, norm_params)
        inputs = tf.constant([norm_input.values], dtype=tf.float32)
        pred = model.predict(inputs, batch_size=1)
        processed_pred = post_process_prediction(pred, data_df, norm_params)

        return jsonify({"prediction": processed_pred.to_dict(orient="records")})

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", "-m", type=str, required=False)
    parser.add_argument("--norm-params-path", "-n", type=str, required=False)
    parser.add_argument("--debug", action="store_true", required=False)
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    model_path = get_model_path()
    if args.model_path:
        model_path = args.model_path

    if not model_path:
        raise ValueError("Model path not provided or set in the environment")

    norm_params_path = get_norm_params_path()
    if args.norm_params_path:
        norm_params_path = args.norm_params_path
    
    if not norm_params_path:
        raise ValueError("Normalization parameters path not provided or set in the environment")

    app = create_app(model_path, norm_params_path, args.debug)

    print(f"Starting server on {args.host}:{args.port}")
    print(f"Model: {model_path}")

    app.run(
        debug=args.debug,
        port=args.port,
        host=args.host,
    )
