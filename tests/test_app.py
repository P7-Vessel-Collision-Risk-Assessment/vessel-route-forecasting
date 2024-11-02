import pytest
import json
import numpy as np
from src.app import create_app
from src.utils import get_model_path


@pytest.fixture
def app():
    model_path = get_model_path()

    if not model_path:
        raise ValueError("Model path not found")

    app = create_app(model_path, debug=True)
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
    data = {"data": np.random.random((1, 20, 3)).tolist()}
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "prediction" in data
