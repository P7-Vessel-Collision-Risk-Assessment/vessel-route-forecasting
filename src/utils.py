import tensorflow as tf
import os
import pandas as pd
import pyproj
import numpy as np


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


def get_norm_params_path() -> str | None:
    return os.environ.get("NORM_PARAMS_PATH", None)


def normalize_inputs(df, norm_params):
    normalized_df = pd.DataFrame()
    for feature in norm_params.columns:
        feature_cols = [col for col in df.columns if col.startswith(feature)]
        for col in feature_cols:
            normalized_df[col] = (
                df[col] - norm_params.loc["sc_x_mean", feature]
            ) / norm_params.loc["sc_x_std", feature]
    return normalized_df


def timestamp_to_unix(timestamp: pd.DataFrame):
    return (timestamp - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")


def wgs84_to_utm(lon, lat, inverse=False):

    # proj_string = f"+proj=utm +zone=32 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
    myProj = pyproj.Proj(
        proj="utm",
        zone=32,
        ellps="WGS84",
        datum="WGS84",
        units="m",
        no_defs=True,
        south=False,
    )

    result_x, result_y = myProj(lon, lat, inverse=inverse)
    return result_x, result_y


def calc_dt_dutmx_dutmy(timestamps, utm_xs, utm_ys):
    dt = timestamps.diff().values
    dutm_x = utm_xs.diff().values
    dutm_y = utm_ys.diff().values

    return dt, dutm_x, dutm_y


def pre_process_data(data: pd.DataFrame, norm_params: pd.DataFrame):
    utm_xs, utm_ys = wgs84_to_utm(data["longitude"].values, data["latitude"].values)
    data["t"] = timestamp_to_unix(data["timestamp"])
    data["utm_x"] = utm_xs
    data["utm_y"] = utm_ys

    # calculate deltas
    dts, dutm_xs, dutm_ys = calc_dt_dutmx_dutmy(data["t"], data["utm_x"], data["utm_y"])
    data["dt"] = dts
    data["dutm_x"] = dutm_xs
    data["dutm_y"] = dutm_ys

    # the first row has no previous row to calculate delta, so we get NaN
    data = data.dropna()

    # normalize inputs
    normalized_inputs = normalize_inputs(data[["dt", "dutm_x", "dutm_y"]], norm_params)

    return normalized_inputs, data


def post_process_prediction(prediction: list, trajectory: pd.DataFrame, norm_param):
    look_ahead_points = 32

    features_outputs = [f"dutm_x(t+{i})" for i in range(1, look_ahead_points + 1)] + [
        f"dutm_y(t+{i})" for i in range(1, look_ahead_points + 1)
    ]
    normalized = pd.DataFrame(prediction, columns=features_outputs)

    # denormalize
    denormalized = pd.DataFrame()

    for feature in norm_param.columns:
        feature_cols = [col for col in normalized.columns if col.startswith(feature)]
        for col in feature_cols:
            denormalized[col] = (
                normalized[col] * norm_param.loc["sc_x_std", feature]
                + norm_param.loc["sc_x_mean", feature]
            )

    # convert to actual values
    for la in range(1, look_ahead_points + 1):
        denormalized[f"utm_x(t+{la})"] = (
            denormalized[f"dutm_x(t+{la})"] + trajectory["utm_x"].iloc[-1]
        )
        denormalized[f"utm_y(t+{la})"] = (
            denormalized[f"dutm_y(t+{la})"] + trajectory["utm_y"].iloc[-1]
        )
        denormalized = denormalized.copy()

    # calculate predicted speed
    if len(trajectory) >= look_ahead_points:
        for la in range(1, look_ahead_points + 1):
            idx = la - 1
            denormalized[f"speed(t+{la})"] = (
                np.sqrt(
                    (denormalized[f"utm_x(t+{la})"] - trajectory["utm_x"].iloc[idx])
                    ** 2
                    + (denormalized[f"utm_y(t+{la})"] - trajectory["utm_y"].iloc[idx])
                    ** 2
                )
                / 60
            )

    # convert from utm to wgs84
    results = {}
    utm_columns = [
        (f"utm_x(t+{i})", f"utm_y(t+{i})") for i in range(1, look_ahead_points + 1)
    ]

    for lon_col, lat_col in utm_columns:
        wgs_lon_col = lon_col.replace("utm_x", "lon")
        wgs_lat_col = lat_col.replace("utm_y", "lat")

        results[wgs_lon_col], results[wgs_lat_col] = zip(
            *denormalized.apply(
                lambda row: wgs84_to_utm(row[lon_col], row[lat_col], inverse=True),
                axis=1,
            )
        )

    df_results = pd.concat([denormalized, pd.DataFrame(results)], axis=1)

    return df_results
