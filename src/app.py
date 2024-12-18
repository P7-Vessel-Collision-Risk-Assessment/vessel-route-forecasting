from datetime import datetime
from io import StringIO
import logging
import pandas as pd
import tensorflow as tf
import argparse
from flask import Flask, request, jsonify

from utils import (
    dist_euclidean,
    get_model_path,
    get_norm_params_path,
    post_process_prediction,
    pre_process_data,
)


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

    @app.route("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.route("/predict", methods=["POST"])
    def predict():
        if not request.json:
            return jsonify({"error": "No data provided"})

        data_json = request.json
        data_df = pd.read_json(StringIO(data_json.get("data")))

        norm_input_list = []
        data_df_list = []
        grouped = data_df.groupby("trajectory_id")
        for _, group in grouped:
            norm_input, data = pre_process_data(group, norm_params)
            norm_input_list.append(norm_input.values)
            data_df_list.append(data)

        inputs = tf.constant(norm_input_list, dtype=tf.float32)

        pred = model.predict(inputs, batch_size=32)

        predictions = pd.DataFrame()
        for i in range(pred.shape[0]):
            processed_pred = post_process_prediction(
                pred[i], pd.DataFrame(data_df_list[i]), norm_params
            )
            predictions = pd.concat([predictions, processed_pred])

        return jsonify({"prediction": predictions.to_dict(orient="records")})

    # Logging
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s: %(message)s")

    @app.before_request
    def log_request_info():
        if request.path != "/health":
            app.logger.info(
                f"Request: {request.remote_addr} {request.method} {request.url}"
            )
            app.logger.debug(f"Headers: {request.headers}")
            app.logger.debug(f"Body: {request.get_data()}")

    @app.after_request
    def log_response_info(response):
        if request.path != "/health":
            app.logger.info(f"Response status: {response.status}")
            app.logger.debug(f"Response headers: {response.headers}")
        return response

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
        raise ValueError(
            "Normalization parameters path not provided or set in the environment"
        )

    print(f"Starting server on {args.host}:{args.port}")
    print(f"Debug mode: {args.debug}")
    print(f"Model: {model_path}")
    print(f"Normalization parameters: {norm_params_path}")
    app = create_app(model_path, norm_params_path, args.debug)

    app.run(host=args.host, port=args.port, debug=args.debug)
