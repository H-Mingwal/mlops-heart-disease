from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_health_endpoint_returns_status():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_metrics_endpoint_returns_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "api_requests_total" in response.text
