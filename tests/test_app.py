import pandas as pd
import pytest
import json
import numpy as np
from src.app import create_app
from src.utils import get_model_path, get_norm_params_path


@pytest.fixture
def app():
    model_path = get_model_path()

    if not model_path:
        raise ValueError("Model path not found")

    norm_params_path = get_norm_params_path()

    if not norm_params_path:
        raise ValueError("Normalization parameters path not found")
    
    app = create_app(model_path, norm_params_path, debug=True)
    return app


@pytest.fixture
def client(app):
    return app.test_client()


def test_info(client):
    response = client.get("/")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "model" in data
    assert "input_shape" in data
    assert "gpus_available" in data
    assert "server_info" in data
    assert "timestamp" in data["server_info"]
    assert "startup_timestamp" in data["server_info"]
    assert "debug_mode" in data["server_info"]


def test_predict(client):
    batch_size = 1  # Testing with a batch of 1
    time_steps = 20
    num_features = 3

    traj_pos = pd.DataFrame({
        "utm_x": np.random.random(time_steps),
        "utm_y": np.random.random(time_steps)
    }).to_json(orient='records')
    data = pd.DataFrame({
        "d": np.random.random(time_steps),
        "dlon": np.random.random(time_steps),
        "dlat": np.random.random(time_steps)
    }).to_json(orient='records')
    request = {"data": data, "trajectory": traj_pos}
    response = client.post("/predict", json=request)
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "prediction" in data
