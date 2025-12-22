import pytest


class TestUserRegistration:
    """Тесты регистрации пользователей"""

    def test_register_new_user(self, client):
        """Тест успешной регистрации нового пользователя"""
        response = client.post(
            "/user/register",
            json={
                "email": "newuser@example.com",
                "password": "securepassword123"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["email"] == "newuser@example.com"
        assert "hashed_password" in data
        assert data["hashed_password"] != "securepassword123"  # Должен быть хеширован

    def test_register_duplicate_email(self, client, test_user):
        """Тест регистрации с существующим email"""
        response = client.post(
            "/user/register",
            json={
                "email": test_user.email,
                "password": "anotherpassword"
            }
        )

        assert response.status_code == 400
        assert "already registered" in response.json()["detail"].lower()

    def test_register_invalid_email(self, client):
        """Тест регистрации с невалидным email"""
        response = client.post(
            "/user/register",
            json={
                "email": "not-an-email",
                "password": "password123"
            }
        )

        assert response.status_code == 422  # Validation error

    def test_register_missing_password(self, client):
        """Тест регистрации без пароля"""
        response = client.post(
            "/user/register",
            json={
                "email": "test@example.com"
            }
        )

        assert response.status_code == 422

    def test_register_missing_email(self, client):
        """Тест регистрации без email"""
        response = client.post(
            "/user/register",
            json={
                "password": "password123"
            }
        )

        assert response.status_code == 422

    def test_register_creates_balance(self, client, db_session):
        """Тест что регистрация создает баланс пользователя"""
        response = client.post(
            "/user/register",
            json={
                "email": "withbalance@example.com",
                "password": "password123"
            }
        )

        assert response.status_code == 200
        user_id = response.json()["id"]

        # Проверяем что баланс создан
        from src.models.balance import Balance
        balance = db_session.query(Balance).filter(
            Balance.user_id == user_id
        ).first()

        assert balance is not None
        assert balance.amount == 100.0  # Начальный баланс

    def test_register_creates_initial_transaction(self, client, db_session):
        """Тест что регистрация создает начальную транзакцию"""
        response = client.post(
            "/user/register",
            json={
                "email": "withtrans@example.com",
                "password": "password123"
            }
        )

        assert response.status_code == 200
        user_id = response.json()["id"]

        # Проверяем что транзакция создана
        from src.models.balance import Transaction
        transaction = db_session.query(Transaction).filter(
            Transaction.user_id == user_id
        ).first()

        assert transaction is not None
        assert transaction.amount == 100.0
        assert transaction.transaction_type == "CREDIT"
        assert transaction.transaction_status == "DONE"


class TestUserLogin:
    """Тесты входа пользователей"""

    def test_login_success(self, client, test_user):
        """Тест успешного входа"""
        response = client.post(
            "/user/login",
            json={
                "email": test_user.email,
                "password": "testpassword123"
            }
        )

        assert response.status_code == 200
        token = response.json()
        assert isinstance(token, str)
        assert len(token) > 0

    def test_login_wrong_password(self, client, test_user):
        """Тест входа с неправильным паролем"""
        response = client.post(
            "/user/login",
            json={
                "email": test_user.email,
                "password": "wrongpassword"
            }
        )

        assert response.status_code == 400
        assert "password" in response.json()["detail"].lower()

    def test_login_nonexistent_user(self, client):
        """Тест входа несуществующего пользователя"""
        response = client.post(
            "/user/login",
            json={
                "email": "nonexistent@example.com",
                "password": "password123"
            }
        )

        assert response.status_code == 400
        assert "not registered" in response.json()["detail"].lower()

    def test_login_invalid_email(self, client):
        """Тест входа с невалидным email"""
        response = client.post(
            "/user/login",
            json={
                "email": "not-an-email",
                "password": "password123"
            }
        )

        assert response.status_code == 422

    def test_login_returns_valid_token(self, client, test_user):
        """Тест что логин возвращает валидный токен"""
        response = client.post(
            "/user/login",
            json={
                "email": test_user.email,
                "password": "testpassword123"
            }
        )

        assert response.status_code == 200
        token = response.json()

        # Используем токен для доступа к защищенному endpoint
        me_response = client.get(
            "/user/me",
            headers={"Authorization": f"Bearer {token}"}
        )

        assert me_response.status_code == 200
        assert me_response.json()["email"] == test_user.email


class TestGetCurrentUser:
    """Тесты получения данных текущего пользователя"""

    def test_get_me_success(self, client, test_user_token, test_user):
        """Тест успешного получения данных"""
        response = client.get(
            "/user/me",
            headers={"Authorization": f"Bearer {test_user_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["email"] == test_user.email
        assert "id" in data
        assert "balance" in data
        assert "transactions" in data

    def test_get_me_without_token(self, client):
        """Тест получения данных без токена"""
        response = client.get("/user/me")

        assert response.status_code == 403

    def test_get_me_with_invalid_token(self, client):
        """Тест получения данных с невалидным токеном"""
        response = client.get(
            "/user/me",
            headers={"Authorization": "Bearer invalid_token"}
        )

        assert response.status_code == 403

    def test_get_me_includes_balance(self, client, test_user_token, test_user):
        """Тест что ответ включает баланс"""
        response = client.get(
            "/user/me",
            headers={"Authorization": f"Bearer {test_user_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        assert "balance" in data
        assert "amount" in data["balance"]
        assert data["balance"]["amount"] == 100.0

    def test_get_me_includes_transactions(self, client, test_user_token):
        """Тест что ответ включает транзакции"""
        response = client.get(
            "/user/me",
            headers={"Authorization": f"Bearer {test_user_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        assert "transactions" in data
        assert isinstance(data["transactions"], list)
        assert len(data["transactions"]) >= 1  # Начальная транзакция

    def test_get_me_structure(self, client, test_user_token, test_user):
        """Тест структуры ответа"""
        response = client.get(
            "/user/me",
            headers={"Authorization": f"Bearer {test_user_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        # Проверяем основные поля
        assert "id" in data
        assert "email" in data
        assert "balance" in data
        assert "transactions" in data

        # Проверяем структуру баланса
        balance = data["balance"]
        assert "id" in balance
        assert "user_id" in balance
        assert "amount" in balance

        # Проверяем структуру транзакций
        if data["transactions"]:
            transaction = data["transactions"][0]
            assert "id" in transaction
            assert "user_id" in transaction
            assert "amount" in transaction
            assert "transaction_type" in transaction
            assert "transaction_status" in transaction
            assert "timestamp" in transaction


class TestPasswordSecurity:
    """Тесты безопасности паролей"""

    def test_password_is_hashed(self, client, db_session):
        """Тест что пароль хешируется"""
        response = client.post(
            "/user/register",
            json={
                "email": "hashtest@example.com",
                "password": "plainpassword123"
            }
        )

        assert response.status_code == 200

        from src.models.user import User
        user = db_session.query(User).filter(
            User.email == "hashtest@example.com"
        ).first()

        assert user is not None
        assert user.hashed_password != "plainpassword123"
        assert len(user.hashed_password) > 50  # Хеш должен быть длинным

    def test_password_not_returned_in_response(self, client, test_user_token):
        """Тест что пароль не возвращается в ответе"""
        response = client.get(
            "/user/me",
            headers={"Authorization": f"Bearer {test_user_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        assert "password" not in data
        assert "hashed_password" not in data