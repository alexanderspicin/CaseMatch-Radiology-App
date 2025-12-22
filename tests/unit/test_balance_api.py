import pytest
from src.models.balance import Balance, Transaction


class TestBalanceCreditEndpoint:
    """Тесты для пополнения баланса"""

    def test_credit_success(self, client, auth_headers, test_user, db_session):
        """Тест успешного пополнения баланса"""
        # Получаем начальный баланс
        initial_balance = db_session.query(Balance).filter(
            Balance.user_id == test_user.id
        ).first()
        initial_amount = initial_balance.amount

        response = client.get(
            "/balance/credit?amount=50",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()

        assert "id" in data
        assert data["transaction_type"] == "CREDIT"
        assert data["amount"] == 50.0
        assert data["transaction_status"] == "DONE"

        # Проверяем что баланс увеличился
        db_session.refresh(initial_balance)
        # 50 рублей * 1.2 (курс) = 60 токенов
        expected_amount = initial_amount + (50 * 1.2)
        assert initial_balance.amount == pytest.approx(expected_amount, rel=0.01)

    def test_credit_without_auth(self, client):
        """Тест пополнения без авторизации"""
        response = client.get("/balance/credit?amount=100")

        assert response.status_code == 403

    def test_credit_negative_amount(self, client, auth_headers):
        """Тест пополнения отрицательной суммой"""
        response = client.get(
            "/balance/credit?amount=-50",
            headers=auth_headers
        )

        assert response.status_code == 400
        assert "greater than zero" in response.json()["detail"].lower()

    def test_credit_zero_amount(self, client, auth_headers):
        """Тест пополнения нулевой суммой"""
        response = client.get(
            "/balance/credit?amount=0",
            headers=auth_headers
        )

        assert response.status_code == 400

    def test_credit_creates_transaction(self, client, auth_headers, test_user, db_session):
        """Тест что пополнение создает транзакцию"""
        # Считаем количество транзакций до
        initial_count = db_session.query(Transaction).filter(
            Transaction.user_id == test_user.id
        ).count()

        response = client.get(
            "/balance/credit?amount=100",
            headers=auth_headers
        )

        assert response.status_code == 200

        # Проверяем что создана новая транзакция
        final_count = db_session.query(Transaction).filter(
            Transaction.user_id == test_user.id
        ).count()

        assert final_count == initial_count + 1

    def test_credit_exchange_rate_applied(self, client, auth_headers, test_user, db_session):
        """Тест что курс обмена применяется правильно"""
        initial_balance = db_session.query(Balance).filter(
            Balance.user_id == test_user.id
        ).first()
        initial_amount = initial_balance.amount

        # Пополняем на 100 рублей
        response = client.get(
            "/balance/credit?amount=100",
            headers=auth_headers
        )

        assert response.status_code == 200

        # Проверяем что баланс увеличился на 100 * 1.2 = 120 токенов
        db_session.refresh(initial_balance)
        expected_amount = initial_amount + 120.0
        assert initial_balance.amount == pytest.approx(expected_amount, rel=0.01)

    def test_credit_large_amount(self, client, auth_headers, test_user, db_session):
        """Тест пополнения большой суммой"""
        response = client.get(
            "/balance/credit?amount=10000",
            headers=auth_headers
        )

        assert response.status_code == 200

        balance = db_session.query(Balance).filter(
            Balance.user_id == test_user.id
        ).first()

        # Должно быть больше 10000 токенов (с учетом курса и начального баланса)
        assert balance.amount > 10000

    def test_credit_multiple_times(self, client, auth_headers, test_user, db_session):
        """Тест множественных пополнений"""
        # Первое пополнение
        response1 = client.get(
            "/balance/credit?amount=50",
            headers=auth_headers
        )
        assert response1.status_code == 200

        # Второе пополнение
        response2 = client.get(
            "/balance/credit?amount=30",
            headers=auth_headers
        )
        assert response2.status_code == 200

        # Проверяем финальный баланс
        balance = db_session.query(Balance).filter(
            Balance.user_id == test_user.id
        ).first()

        # Начальный (100) + 50*1.2 (60) + 30*1.2 (36) = 196
        expected = 100 + 60 + 36
        assert balance.amount == pytest.approx(expected, rel=0.01)


class TestTransactionService:
    """Тесты для сервиса транзакций"""

    def test_transaction_has_correct_fields(self, client, auth_headers):
        """Тест что транзакция имеет все необходимые поля"""
        response = client.get(
            "/balance/credit?amount=100",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()

        required_fields = [
            "id",
            "user_id",
            "amount",
            "timestamp",
            "transaction_type",
            "transaction_status"
        ]

        for field in required_fields:
            assert field in data

    def test_transaction_timestamp(self, client, auth_headers):
        """Тест что транзакция имеет timestamp"""
        response = client.get(
            "/balance/credit?amount=100",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()

        assert "timestamp" in data
        assert data["timestamp"] is not None
        # Проверяем формат ISO 8601
        from datetime import datetime
        datetime.fromisoformat(data["timestamp"].replace('Z', '+00:00'))

    def test_debit_transaction_insufficient_balance(self, client, auth_headers, test_user, db_session):
        """Тест дебетовой транзакции при недостаточном балансе"""
        from src.services.transaction_service import create_transaction, process_transaction
        from src.schemas.balance import CreateTransactionSchema

        # Пытаемся снять больше чем есть
        transaction_data = CreateTransactionSchema(
            user_id=test_user.id,
            amount=1000.0,  # Больше чем начальные 100
            transaction_type="DEBIT"
        )

        db_transaction = create_transaction(transaction_data, db=db_session)
        result = process_transaction(db_transaction.id, db=db_session)

        assert result.transaction_status == "FAILED"

    def test_debit_transaction_success(self, client, auth_headers, test_user, db_session):
        """Тест успешной дебетовой транзакции"""
        from src.services.transaction_service import create_transaction, process_transaction
        from src.schemas.balance import CreateTransactionSchema

        initial_balance = db_session.query(Balance).filter(
            Balance.user_id == test_user.id
        ).first()
        initial_amount = initial_balance.amount

        # Снимаем 50 токенов
        transaction_data = CreateTransactionSchema(
            user_id=test_user.id,
            amount=50.0,
            transaction_type="DEBIT"
        )

        db_transaction = create_transaction(transaction_data, db=db_session)
        result = process_transaction(db_transaction.id, db=db_session)

        assert result.transaction_status == "DONE"

        # Проверяем что баланс уменьшился
        db_session.refresh(initial_balance)
        assert initial_balance.amount == initial_amount - 50.0


class TestExchangeService:
    """Тесты для сервиса обмена"""

    def test_exchange_service_exists(self, db_session):
        """Тест что сервис обмена существует"""
        from src.models.balance import ExchangeService

        exchange_service = db_session.query(ExchangeService).first()
        assert exchange_service is not None
        assert exchange_service.rate > 0

    def test_exchange_rate_conversion(self, db_session):
        """Тест конвертации валюты"""
        from src.services.exchange_service import convert_rub_to_tokens

        # 100 рублей должно быть 120 токенов при курсе 1.2
        tokens = convert_rub_to_tokens(100.0, db=db_session)
        assert tokens == 120.0

    def test_exchange_rate_various_amounts(self, db_session):
        """Тест конвертации различных сумм"""
        from src.services.exchange_service import convert_rub_to_tokens

        test_cases = [
            (100, 120),
            (50, 60),
            (1, 1.2),
            (1000, 1200),
        ]

        for rubles, expected_tokens in test_cases:
            tokens = convert_rub_to_tokens(rubles, db=db_session)
            assert tokens == pytest.approx(expected_tokens, rel=0.01)


class TestBalanceIntegration:
    """Интеграционные тесты баланса"""

    def test_user_balance_lifecycle(self, client, db_session):
        """Тест полного жизненного цикла баланса пользователя"""
        # 1. Регистрация
        response = client.post(
            "/user/register",
            json={
                "email": "lifecycle@example.com",
                "password": "password123"
            }
        )
        assert response.status_code == 200
        user_id = response.json()["id"]

        # 2. Проверка начального баланса
        balance = db_session.query(Balance).filter(
            Balance.user_id == user_id
        ).first()
        assert balance.amount == 100.0

        # 3. Вход
        login_response = client.post(
            "/user/login",
            json={
                "email": "lifecycle@example.com",
                "password": "password123"
            }
        )
        token = login_response.json()
        headers = {"Authorization": f"Bearer {token}"}

        # 4. Пополнение
        credit_response = client.get(
            "/balance/credit?amount=100",
            headers=headers
        )
        assert credit_response.status_code == 200

        # 5. Проверка увеличения баланса
        db_session.refresh(balance)
        # 100 + (100 * 1.2) = 220
        assert balance.amount == 220.0

        # 6. Проверка истории транзакций
        me_response = client.get("/user/me", headers=headers)
        user_data = me_response.json()

        # Должно быть минимум 2 транзакции (начальная + пополнение)
        assert len(user_data["transactions"]) >= 2

    def test_concurrent_credits(self, client, auth_headers, test_user, db_session):
        """Тест конкурентных пополнений"""
        # Множественные пополнения подряд
        amounts = [10, 20, 30, 40, 50]

        for amount in amounts:
            response = client.get(
                f"/balance/credit?amount={amount}",
                headers=auth_headers
            )
            assert response.status_code == 200

        # Проверяем итоговый баланс
        balance = db_session.query(Balance).filter(
            Balance.user_id == test_user.id
        ).first()

        # 100 (начальный) + (10+20+30+40+50)*1.2 = 100 + 180 = 280
        expected = 100 + (sum(amounts) * 1.2)
        assert balance.amount == pytest.approx(expected, rel=0.01)