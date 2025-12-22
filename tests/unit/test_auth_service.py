import pytest
from datetime import timedelta
from jose import jwt

from src.services.auth_service import (
    create_access_token,
    decode_jwt,
    JWTBearer
)
from src.core.config import settings


class TestAuthService:
    """Тесты для сервиса аутентификации"""

    def test_create_access_token(self):
        """Тест создания JWT токена"""
        data = {"email": "test@example.com"}
        token = create_access_token(data)

        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0

    def test_create_access_token_with_expiration(self):
        """Тест создания токена с кастомным временем жизни"""
        data = {"email": "test@example.com"}
        expires_delta = timedelta(minutes=15)
        token = create_access_token(data, expires_delta)

        decoded = decode_jwt(token)
        assert "email" in decoded
        assert decoded["email"] == "test@example.com"
        assert "exp" in decoded

    def test_decode_jwt_valid(self):
        """Тест декодирования валидного токена"""
        data = {"email": "test@example.com", "sub": "123"}
        token = create_access_token(data)

        decoded = decode_jwt(token)

        assert decoded["email"] == "test@example.com"
        assert decoded["sub"] == "123"
        assert "exp" in decoded

    def test_decode_jwt_invalid(self):
        """Тест декодирования невалидного токена"""
        invalid_token = "invalid.token.here"
        decoded = decode_jwt(invalid_token)

        assert decoded == {}

    def test_decode_jwt_expired(self):
        """Тест декодирования истекшего токена"""
        data = {"email": "test@example.com"}
        # Создаем токен с отрицательным временем жизни
        token = create_access_token(data, timedelta(seconds=-1))

        decoded = decode_jwt(token)
        assert decoded == {}

    def test_jwt_bearer_invalid_scheme(self, client):
        """Тест с неправильной схемой авторизации"""
        response = client.get(
            "/user/me",
            headers={"Authorization": "Basic invalid_token"}
        )
        assert response.status_code == 403

    def test_jwt_bearer_no_token(self, client):
        """Тест без токена"""
        response = client.get("/user/me")
        assert response.status_code == 403

    def test_token_contains_correct_algorithm(self):
        """Тест что токен использует правильный алгоритм"""
        data = {"email": "test@example.com"}
        token = create_access_token(data)

        # Декодируем без верификации чтобы проверить header
        unverified = jwt.get_unverified_header(token)
        assert unverified["alg"] == settings.algorithm

    def test_token_payload_structure(self):
        """Тест структуры payload токена"""
        data = {
            "email": "test@example.com",
            "user_id": "123",
            "role": "user"
        }
        token = create_access_token(data)
        decoded = decode_jwt(token)

        assert "email" in decoded
        assert "user_id" in decoded
        assert "role" in decoded
        assert "exp" in decoded

        assert decoded["email"] == data["email"]
        assert decoded["user_id"] == data["user_id"]
        assert decoded["role"] == data["role"]


class TestJWTBearerIntegration:
    """Интеграционные тесты для JWT Bearer"""

    def test_valid_token_authentication(self, client, test_user_token, test_user):
        """Тест аутентификации с валидным токеном"""
        response = client.get(
            "/user/me",
            headers={"Authorization": f"Bearer {test_user_token}"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["email"] == test_user.email

    def test_expired_token_authentication(self, client):
        """Тест аутентификации с истекшим токеном"""
        expired_token = create_access_token(
            {"email": "test@example.com"},
            timedelta(seconds=-1)
        )

        response = client.get(
            "/user/me",
            headers={"Authorization": f"Bearer {expired_token}"}
        )

        assert response.status_code == 403

    def test_malformed_token(self, client):
        """Тест с искаженным токеном"""
        response = client.get(
            "/user/me",
            headers={"Authorization": "Bearer malformed.token.here"}
        )

        assert response.status_code == 403

    def test_token_with_nonexistent_user(self, client):
        """Тест с токеном для несуществующего пользователя"""
        token = create_access_token({"email": "nonexistent@example.com"})

        response = client.get(
            "/user/me",
            headers={"Authorization": f"Bearer {token}"}
        )

        assert response.status_code == 401