import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient


VENDOR_ROOT = Path(__file__).resolve().parents[1]
if str(VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_ROOT))

from API_servers.router.v1_routes import router


def test_batch_translate_requires_password_when_configured():
    app = FastAPI()
    app.include_router(router)
    app.state.password = "secret"
    app.state.engine = object()

    with TestClient(app) as client:
        response = client.post(
            "/translate/v1/batch-translate",
            json={
                "source_lang": "en",
                "target_lang": "zh-CN",
                "text_list": ["hello"],
            },
        )

    assert response.status_code == 401
    assert response.json() == {"error": "Unauthorized: invalid or missing password"}
