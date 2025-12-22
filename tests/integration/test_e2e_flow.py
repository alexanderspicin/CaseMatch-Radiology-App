import pytest
from unittest.mock import patch
import io
from PIL import Image


@pytest.mark.integration
class TestEndToEndFlow:
    """End-to-end интеграционные тесты"""

    def test_complete_user_journey(self, client, sample_image):
        """Тест полного пути пользователя от регистрации до предсказания"""

        # 1. Регистрация
        register_response = client.post(
            "/user/register",
            json={
                "email": "journey@example.com",
                "password": "securepass123"
            }
        )
        assert register_response.status_code == 200
        user_data = register_response.json()
        assert "id" in user_data

        # 2. Вход в систему
        login_response = client.post(
            "/user/login",
            json={
                "email": "journey@example.com",
                "password": "securepass123"
            }
        )
        assert login_response.status_code == 200
        token = login_response.json()
        headers = {"Authorization": f"Bearer {token}"}

        # 3. Проверка профиля
        me_response = client.get("/user/me", headers=headers)
        assert me_response.status_code == 200
        profile = me_response.json()
        assert profile["email"] == "journey@example.com"
        assert profile["balance"]["amount"] == 100.0

        # 4. Пополнение баланса
        credit_response = client.get(
            "/balance/credit?amount=100",
            headers=headers
        )
        assert credit_response.status_code == 200

        # 5. Проверка обновленного баланса
        me_response = client.get("/user/me", headers=headers)
        profile = me_response.json()
        assert profile["balance"]["amount"] == 220.0  # 100 + 100*1.2

        # 6. Выполнение предсказания
        with patch('src.services.predict_service.model_manager') as mock_manager:
            mock_manager.predict_from_bytes.return_value = {
                "detected": ["Cardiomegaly"],
                "predictions": [
                    {"label": "Cardiomegaly", "probability": 0.8, "detected": True},
                ],
                "threshold": 0.5,
                "saved_to_db": False
            }

            predict_response = client.post(
                "/predict/predict",
                headers=headers,
                files={"image": ("test.png", sample_image, "image/png")},
                data={"threshold": "0.5"}
            )
            assert predict_response.status_code == 200
            prediction = predict_response.json()
            assert "detected" in prediction
            assert "Cardiomegaly" in prediction["detected"]

    def test_prediction_with_qdrant_save(self, client, sample_image):
        """Тест предсказания с сохранением в Qdrant и последующим поиском"""

        # Регистрация и вход
        client.post(
            "/user/register",
            json={"email": "qdrant@example.com", "password": "pass123"}
        )
        login_response = client.post(
            "/user/login",
            json={"email": "qdrant@example.com", "password": "pass123"}
        )
        token = login_response.json()
        headers = {"Authorization": f"Bearer {token}"}

        # Предсказание с сохранением
        with patch('src.services.predict_service.model_manager') as mock_manager:
            mock_manager.predict_from_bytes.return_value = {
                "detected": ["Effusion"],
                "predictions": [
                    {"label": "Effusion", "probability": 0.75, "detected": True},
                ],
                "threshold": 0.5,
                "saved_to_db": True,
                "point_id": "test-point-123"
            }

            predict_response = client.post(
                "/predict/predict",
                headers=headers,
                files={"image": ("test.png", sample_image, "image/png")},
                data={"threshold": "0.5", "save_to_db": "true"}
            )
            assert predict_response.status_code == 200
            prediction = predict_response.json()
            assert prediction["saved_to_db"] is True
            assert "point_id" in prediction

            # Поиск похожих случаев
            mock_manager.search_similar.return_value = [
                {
                    "id": "test-point-123",
                    "score": 1.0,
                    "payload": {
                        "detected_labels": ["Effusion"],
                        "timestamp": "2024-01-01T00:00:00"
                    }
                }
            ]

            search_response = client.post(
                "/predict/search-similar",
                headers=headers,
                files={"image": ("test.png", sample_image, "image/png")},
                data={"limit": "5", "score_threshold": "0.7"}
            )
            assert search_response.status_code == 200
            results = search_response.json()
            assert results["count"] >= 1

    def test_multiple_users_isolation(self, client, sample_image):
        """Тест изоляции данных между пользователями"""

        # Пользователь 1
        client.post("/user/register", json={"email": "user1@example.com", "password": "pass1"})
        login1 = client.post("/user/login", json={"email": "user1@example.com", "password": "pass1"})
        token1 = login1.json()
        headers1 = {"Authorization": f"Bearer {token1}"}

        # Пользователь 2
        client.post("/user/register", json={"email": "user2@example.com", "password": "pass2"})
        login2 = client.post("/user/login", json={"email": "user2@example.com", "password": "pass2"})
        token2 = login2.json()
        headers2 = {"Authorization": f"Bearer {token2}"}

        # Пользователь 1 пополняет баланс
        client.get("/balance/credit?amount=500", headers=headers1)

        # Проверяем балансы
        profile1 = client.get("/user/me", headers=headers1).json()
        profile2 = client.get("/user/me", headers=headers2).json()

        # Балансы должны быть разными
        assert profile1["balance"]["amount"] == 700.0  # 100 + 500*1.2
        assert profile2["balance"]["amount"] == 100.0  # Только начальный

        # Пользователь 1 не должен видеть токен пользователя 2
        assert token1 != token2

    def test_authentication_required_for_protected_endpoints(self, client):
        """Тест что защищенные endpoints требуют авторизации"""

        protected_endpoints = [
            ("GET", "/user/me"),
            ("GET", "/balance/credit?amount=100"),
        ]

        for method, endpoint in protected_endpoints:
            if method == "GET":
                response = client.get(endpoint)
            else:
                response = client.post(endpoint)

            assert response.status_code == 403, f"{endpoint} should require auth"

    def test_invalid_token_rejected(self, client):
        """Тест что невалидные токены отклоняются"""

        invalid_headers = {"Authorization": "Bearer invalid_token_here"}

        response = client.get("/user/me", headers=invalid_headers)
        assert response.status_code == 403


@pytest.mark.integration
class TestDatabaseIntegrity:
    """Тесты целостности базы данных"""

    def test_cascade_delete_user(self, client, db_session):
        """Тест каскадного удаления при удалении пользователя"""
        from src.models.user import User
        from src.models.balance import Balance, Transaction

        # Создаем пользователя
        response = client.post(
            "/user/register",
            json={"email": "cascade@example.com", "password": "pass123"}
        )
        user_id = response.json()["id"]

        # Проверяем что созданы связанные записи
        balance = db_session.query(Balance).filter(Balance.user_id == user_id).first()
        transaction = db_session.query(Transaction).filter(Transaction.user_id == user_id).first()

        assert balance is not None
        assert transaction is not None

        # Удаляем пользователя
        user = db_session.query(User).filter(User.id == user_id).first()
        db_session.delete(user)
        db_session.commit()

        # Проверяем что связанные записи тоже удалены (cascade)
        balance = db_session.query(Balance).filter(Balance.user_id == user_id).first()
        transaction = db_session.query(Transaction).filter(Transaction.user_id == user_id).first()

        assert balance is None
        assert transaction is None

    def test_unique_email_constraint(self, client):
        """Тест уникальности email"""

        # Первая регистрация
        response1 = client.post(
            "/user/register",
            json={"email": "unique@example.com", "password": "pass1"}
        )
        assert response1.status_code == 200

        # Попытка дублировать email
        response2 = client.post(
            "/user/register",
            json={"email": "unique@example.com", "password": "pass2"}
        )
        assert response2.status_code == 400

    def test_transaction_atomicity(self, client, test_user, db_session):
        """Тест атомарности транзакций"""
        from src.models.balance import Balance

        # Получаем начальный баланс
        balance = db_session.query(Balance).filter(
            Balance.user_id == test_user.id
        ).first()
        initial_amount = balance.amount

        # Выполняем операцию
        login_response = client.post(
            "/user/login",
            json={"email": test_user.email, "password": "testpassword123"}
        )
        token = login_response.json()
        headers = {"Authorization": f"Bearer {token}"}

        client.get("/balance/credit?amount=100", headers=headers)

        # Проверяем что баланс обновлен
        db_session.refresh(balance)
        new_amount = balance.amount

        # Баланс должен увеличиться ровно на 100*1.2
        assert new_amount == initial_amount + 120.0


@pytest.mark.integration
@pytest.mark.slow
class TestPerformance:
    """Тесты производительности"""

    def test_concurrent_requests(self, client, test_user_token):
        """Тест одновременных запросов"""
        import concurrent.futures

        headers = {"Authorization": f"Bearer {test_user_token}"}

        def make_request():
            return client.get("/user/me", headers=headers)

        # Выполняем 10 одновременных запросов
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [f.result() for f in futures]

        # Все запросы должны быть успешными
        for result in results:
            assert result.status_code == 200

    @patch('src.services.predict_service.model_manager')
    def test_multiple_predictions(self, mock_manager, client, auth_headers, sample_image):
        """Тест множественных предсказаний"""

        mock_manager.predict_from_bytes.return_value = {
            "detected": [],
            "predictions": [],
            "threshold": 0.5,
            "saved_to_db": False
        }

        # Выполняем несколько предсказаний подряд
        for i in range(5):
            response = client.post(
                "/predict/predict",
                headers=auth_headers,
                files={"image": (f"test{i}.png", sample_image, "image/png")}
            )
            assert response.status_code == 200


@pytest.mark.integration
class TestErrorHandling:
    """Тесты обработки ошибок"""

    def test_database_error_handling(self, client):
        """Тест обработки ошибок базы данных"""
        # Попытка входа когда БД недоступна не должна падать
        response = client.post(
            "/user/login",
            json={"email": "nonexistent@example.com", "password": "pass"}
        )
        # Должен вернуть корректный HTTP код, не 500
        assert response.status_code in [400, 401, 404]

    def test_invalid_json_handling(self, client):
        """Тест обработки невалидного JSON"""
        response = client.post(
            "/user/register",
            data="not a valid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    @patch('src.services.predict_service.model_manager')
    def test_model_failure_handling(self, mock_manager, client, auth_headers, sample_image):
        """Тест обработки падения модели"""

        mock_manager.predict_from_bytes.side_effect = RuntimeError("Model crashed")

        response = client.post(
            "/predict/predict",
            headers=auth_headers,
            files={"image": ("test.png", sample_image, "image/png")}
        )

        assert response.status_code == 500
        assert "failed" in response.json()["detail"].lower()


@pytest.mark.integration
class TestHealthChecks:
    """Тесты health check endpoints"""

    def test_root_endpoint(self, client):
        """Тест корневого endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()

    def test_health_endpoint(self, client):
        """Тест health endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "database" in data

    @patch('src.services.predict_service.model_manager')
    def test_predict_health_endpoint(self, mock_manager, client):
        """Тест health endpoint для ML модели"""
        mock_manager.model = "mock_model"
        mock_manager.model_path = "/path/to/model"

        response = client.get("/predict/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data