import pytest
import io
from PIL import Image
from unittest.mock import patch, MagicMock


class TestPredictEndpoint:
    """Тесты для endpoint предсказаний"""

    @patch('src.services.predict_service.model_manager')
    def test_predict_success(self, mock_manager, client, auth_headers, sample_image):
        """Тест успешного предсказания"""
        # Мокаем ответ модели
        mock_manager.predict_from_bytes.return_value = {
            "detected": ["Cardiomegaly"],
            "predictions": [
                {"label": "Cardiomegaly", "probability": 0.8, "detected": True},
                {"label": "Effusion", "probability": 0.3, "detected": False},
            ],
            "threshold": 0.5,
            "saved_to_db": False
        }

        response = client.post(
            "/predict/predict",
            headers=auth_headers,
            files={"image": ("test.png", sample_image, "image/png")},
            data={"threshold": "0.5", "save_to_db": "false"}
        )

        assert response.status_code == 200
        data = response.json()

        assert "detected" in data
        assert "predictions" in data
        assert "threshold" in data
        assert isinstance(data["detected"], list)
        assert isinstance(data["predictions"], list)

    def test_predict_without_auth(self, client, sample_image):
        """Тест предсказания без авторизации"""
        response = client.post(
            "/predict/predict",
            files={"image": ("test.png", sample_image, "image/png")}
        )

        assert response.status_code == 403

    def test_predict_invalid_file_type(self, client, auth_headers):
        """Тест с неподдерживаемым типом файла"""
        text_file = io.BytesIO(b"This is not an image")

        response = client.post(
            "/predict/predict",
            headers=auth_headers,
            files={"image": ("test.txt", text_file, "text/plain")}
        )

        assert response.status_code == 400
        assert "Invalid file type" in response.json()["detail"]

    def test_predict_no_file(self, client, auth_headers):
        """Тест без файла"""
        response = client.post(
            "/predict/predict",
            headers=auth_headers,
            data={"threshold": "0.5"}
        )

        assert response.status_code == 422  # Missing required field

    @patch('src.services.predict_service.model_manager')
    def test_predict_with_custom_threshold(self, mock_manager, client, auth_headers, sample_image):
        """Тест с кастомным порогом"""
        mock_manager.predict_from_bytes.return_value = {
            "detected": ["Cardiomegaly"],
            "predictions": [
                {"label": "Cardiomegaly", "probability": 0.65, "detected": True},
            ],
            "threshold": 0.6,
            "saved_to_db": False
        }

        response = client.post(
            "/predict/predict",
            headers=auth_headers,
            files={"image": ("test.png", sample_image, "image/png")},
            data={"threshold": "0.6"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["threshold"] == 0.6

        # Проверяем что метод вызван с правильным threshold
        mock_manager.predict_from_bytes.assert_called_once()
        call_kwargs = mock_manager.predict_from_bytes.call_args[1]
        assert call_kwargs["threshold"] == 0.6

    @patch('src.services.predict_service.model_manager')
    def test_predict_save_to_db_true(self, mock_manager, client, auth_headers, sample_image, test_user):
        """Тест сохранения в БД"""
        mock_manager.predict_from_bytes.return_value = {
            "detected": ["Cardiomegaly"],
            "predictions": [
                {"label": "Cardiomegaly", "probability": 0.8, "detected": True},
            ],
            "threshold": 0.5,
            "saved_to_db": True,
            "point_id": "test-point-id-123"
        }

        response = client.post(
            "/predict/predict",
            headers=auth_headers,
            files={"image": ("test.png", sample_image, "image/png")},
            data={"threshold": "0.5", "save_to_db": "true"}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["saved_to_db"] is True
        assert "point_id" in data

        # Проверяем что метод вызван с save_to_db=True
        call_kwargs = mock_manager.predict_from_bytes.call_args[1]
        assert call_kwargs["save_to_db"] is True
        assert call_kwargs["user_id"] == str(test_user.id)

    @patch('src.services.predict_service.model_manager')
    def test_predict_returns_correct_structure(self, mock_manager, client, auth_headers, sample_image):
        """Тест структуры ответа"""
        mock_manager.predict_from_bytes.return_value = {
            "detected": ["Cardiomegaly", "Effusion"],
            "predictions": [
                {"label": "Cardiomegaly", "probability": 0.8, "detected": True},
                {"label": "Effusion", "probability": 0.7, "detected": True},
                {"label": "Mass", "probability": 0.3, "detected": False},
            ],
            "threshold": 0.5,
            "saved_to_db": False
        }

        response = client.post(
            "/predict/predict",
            headers=auth_headers,
            files={"image": ("test.png", sample_image, "image/png")},
            data={"threshold": "0.5"}
        )

        assert response.status_code == 200
        data = response.json()

        # Проверяем основные поля
        assert "detected" in data
        assert "predictions" in data
        assert "threshold" in data

        # Проверяем типы
        assert isinstance(data["detected"], list)
        assert isinstance(data["predictions"], list)
        assert isinstance(data["threshold"], float)

        # Проверяем структуру predictions
        for pred in data["predictions"]:
            assert "label" in pred
            assert "probability" in pred
            assert "detected" in pred
            assert isinstance(pred["label"], str)
            assert isinstance(pred["probability"], float)
            assert isinstance(pred["detected"], bool)

    @patch('src.services.predict_service.model_manager')
    def test_predict_model_error_handling(self, mock_manager, client, auth_headers, sample_image):
        """Тест обработки ошибок модели"""
        mock_manager.predict_from_bytes.side_effect = Exception("Model prediction failed")

        response = client.post(
            "/predict/predict",
            headers=auth_headers,
            files={"image": ("test.png", sample_image, "image/png")}
        )

        assert response.status_code == 500
        assert "failed" in response.json()["detail"].lower()


class TestSearchSimilarEndpoint:
    """Тесты для поиска похожих случаев"""

    @patch('src.services.predict_service.model_manager')
    def test_search_similar_success(self, mock_manager, client, auth_headers, sample_image):
        """Тест успешного поиска похожих случаев"""
        mock_manager.search_similar.return_value = [
            {
                "id": "id-1",
                "score": 0.95,
                "payload": {
                    "detected_labels": ["Cardiomegaly"],
                    "timestamp": "2024-01-01T00:00:00",
                    "user_email": "test@example.com"
                }
            },
            {
                "id": "id-2",
                "score": 0.88,
                "payload": {
                    "detected_labels": ["Effusion"],
                    "timestamp": "2024-01-02T00:00:00",
                    "user_email": "test2@example.com"
                }
            }
        ]

        response = client.post(
            "/predict/search-similar",
            headers=auth_headers,
            files={"image": ("test.png", sample_image, "image/png")},
            data={"limit": "5", "score_threshold": "0.7"}
        )

        assert response.status_code == 200
        data = response.json()

        assert "results" in data
        assert "count" in data
        assert data["count"] == 2
        assert len(data["results"]) == 2

    def test_search_similar_without_auth(self, client, sample_image):
        """Тест поиска без авторизации"""
        response = client.post(
            "/predict/search-similar",
            files={"image": ("test.png", sample_image, "image/png")}
        )

        assert response.status_code == 403

    @patch('src.services.predict_service.model_manager')
    def test_search_similar_no_results(self, mock_manager, client, auth_headers, sample_image):
        """Тест когда нет похожих случаев"""
        mock_manager.search_similar.return_value = []

        response = client.post(
            "/predict/search-similar",
            headers=auth_headers,
            files={"image": ("test.png", sample_image, "image/png")},
            data={"limit": "5", "score_threshold": "0.9"}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["count"] == 0
        assert len(data["results"]) == 0

    @patch('src.services.predict_service.model_manager')
    def test_search_similar_custom_params(self, mock_manager, client, auth_headers, sample_image):
        """Тест с кастомными параметрами"""
        mock_manager.search_similar.return_value = []

        response = client.post(
            "/predict/search-similar",
            headers=auth_headers,
            files={"image": ("test.png", sample_image, "image/png")},
            data={"limit": "10", "score_threshold": "0.85"}
        )

        assert response.status_code == 200

        # Проверяем что метод вызван с правильными параметрами
        mock_manager.search_similar.assert_called_once()
        call_args = mock_manager.search_similar.call_args
        assert call_args[1]["limit"] == 10
        assert call_args[1]["score_threshold"] == 0.85


class TestHealthEndpoint:
    """Тесты для health check endpoint"""

    @patch('src.services.predict_service.model_manager')
    def test_health_check_model_loaded(self, mock_manager, client):
        """Тест health check когда модель загружена"""
        mock_manager.model = MagicMock()  # Модель загружена
        mock_manager.model_path = "/app/models/model.h5"

        response = client.get("/predict/health")

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert "model_loaded" in data
        assert "model_path" in data
        assert "timestamp" in data

        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    @patch('src.services.predict_service.model_manager')
    def test_health_check_model_not_loaded(self, mock_manager, client):
        """Тест health check когда модель не загружена"""
        mock_manager.model = None
        mock_manager.model_path = "/app/models/model.h5"

        response = client.get("/predict/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "unhealthy"
        assert data["model_loaded"] is False


class TestImageValidation:
    """Тесты валидации изображений"""

    def test_jpeg_image_accepted(self, client, auth_headers):
        """Тест что JPEG принимается"""
        img = Image.new('RGB', (100, 100), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)

        with patch('src.services.predict_service.model_manager') as mock_manager:
            mock_manager.predict_from_bytes.return_value = {
                "detected": [],
                "predictions": [],
                "threshold": 0.5,
                "saved_to_db": False
            }

            response = client.post(
                "/predict/predict",
                headers=auth_headers,
                files={"image": ("test.jpg", img_bytes, "image/jpeg")}
            )

            assert response.status_code == 200

    def test_png_image_accepted(self, client, auth_headers):
        """Тест что PNG принимается"""
        img = Image.new('RGB', (100, 100), color='blue')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        with patch('src.services.predict_service.model_manager') as mock_manager:
            mock_manager.predict_from_bytes.return_value = {
                "detected": [],
                "predictions": [],
                "threshold": 0.5,
                "saved_to_db": False
            }

            response = client.post(
                "/predict/predict",
                headers=auth_headers,
                files={"image": ("test.png", img_bytes, "image/png")}
            )

            assert response.status_code == 200

    def test_gif_image_rejected(self, client, auth_headers):
        """Тест что GIF отклоняется"""
        img = Image.new('RGB', (100, 100), color='green')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='GIF')
        img_bytes.seek(0)

        response = client.post(
            "/predict/predict",
            headers=auth_headers,
            files={"image": ("test.gif", img_bytes, "image/gif")}
        )

        assert response.status_code == 400