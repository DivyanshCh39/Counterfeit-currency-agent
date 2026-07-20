"""
Router-level tests using FastAPI's TestClient.
These exercise HTTP status codes / wiring, not analysis accuracy.
"""

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _dummy_jpeg_bytes(width=200, height=100) -> bytes:
    image = np.full((height, width, 3), 200, dtype="uint8")
    ok, buf = cv2.imencode(".jpg", image)
    assert ok
    return buf.tobytes()


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_root_lists_ui_link():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["ui"] == "/ui"


def test_web_ui_serves_html():
    response = client.get("/ui")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_upload_rejects_unsupported_extension():
    files = {"file": ("note.bmp", _dummy_jpeg_bytes(), "image/bmp")}
    response = client.post("/upload", files=files)
    assert response.status_code == 400


def test_upload_accepts_supported_image_and_returns_file_id():
    files = {"file": ("note.jpg", _dummy_jpeg_bytes(), "image/jpeg")}
    response = client.post("/upload", files=files)
    assert response.status_code == 200

    data = response.json()
    assert "file_id" in data
    assert data["filename"] == "note.jpg"
    assert data["size_bytes"] > 0


def test_analyze_with_direct_upload_returns_full_response_shape():
    files = {"file": ("note.jpg", _dummy_jpeg_bytes(), "image/jpeg")}
    response = client.post("/analyze", files=files)
    assert response.status_code == 200

    data = response.json()
    assert data["verdict"] in ("likely genuine", "suspicious", "unclear")
    assert "feature_scores" in data
    assert "explanations" in data
    assert "notes" in data


def test_analyze_by_unknown_file_id_returns_404():
    response = client.post("/analyze/does-not-exist.jpg")
    assert response.status_code == 404


def test_upload_then_analyze_by_file_id_round_trip():
    files = {"file": ("note.jpg", _dummy_jpeg_bytes(), "image/jpeg")}
    upload_response = client.post("/upload", files=files)
    assert upload_response.status_code == 200
    file_id = upload_response.json()["file_id"]

    analyze_response = client.post(f"/analyze/{file_id}")
    assert analyze_response.status_code == 200
    data = analyze_response.json()
    assert data["verdict"] in ("likely genuine", "suspicious", "unclear")


def test_analyze_by_file_id_rejects_path_traversal():
    response = client.post("/analyze/..%2F..%2Fetc%2Fpasswd")
    assert response.status_code in (400, 404)
