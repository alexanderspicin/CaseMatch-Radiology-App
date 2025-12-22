import pytest
import os
from typing import Generator
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
import numpy as np
from PIL import Image
import io

from src.api.main import app
from src.db.database import Base, get_db
from src.models.user import User
from src.models.balance import Balance, Transaction, ExchangeService
from src.services.user_service import get_password_hash
from src.services.auth_service import create_access_token

# Тестовая база данных в памяти
SQLALCHEMY_TEST_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="function")
def db_session() -> Generator[Session, None, None]:
    """Создает чистую БД для каждого теста"""
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()

    # Создаем ExchangeService по умолчанию
    exchange_service = ExchangeService(
        type="default",
        rate=1.2
    )
    session.add(exchange_service)
    session.commit()

    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(db_session: Session) -> Generator[TestClient, None, None]:
    """Тестовый клиент FastAPI"""

    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture(scope="function")
def test_user(db_session: Session) -> User:
    """Создает тестового пользователя"""
    user = User(
        email="test@example.com",
        hashed_password=get_password_hash("testpassword123")
    )
    db_session.add(user)
    db_session.flush()

    # Создаем баланс
    balance = Balance(user_id=user.id, amount=100.0)
    db_session.add(balance)

    # Создаем начальную транзакцию
    transaction = Transaction(
        user_id=user.id,
        amount=100.0,
        transaction_type="CREDIT",
        transaction_status="DONE"
    )
    db_session.add(transaction)

    db_session.commit()
    db_session.refresh(user)

    return user


@pytest.fixture(scope="function")
def test_user_token(test_user: User) -> str:
    """Создает JWT токен для тестового пользователя"""
    token = create_access_token({"email": test_user.email})
    return token


@pytest.fixture(scope="function")
def auth_headers(test_user_token: str) -> dict:
    """Заголовки с авторизацией"""
    return {"Authorization": f"Bearer {test_user_token}"}


@pytest.fixture(scope="function")
def sample_image() -> bytes:
    """Создает тестовое изображение"""
    img = Image.new('RGB', (480, 480), color='gray')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr.getvalue()


@pytest.fixture(scope="function")
def sample_xray_image() -> bytes:
    """Создает тестовое рентгеновское изображение с реалистичным шумом"""
    # Создаем изображение с градиентом и шумом для имитации рентгена
    img_array = np.random.randint(0, 256, (480, 480, 3), dtype=np.uint8)

    # Добавляем градиент
    for i in range(480):
        img_array[i, :, :] = img_array[i, :, :] * (i / 480)

    img = Image.fromarray(img_array.astype('uint8'), 'RGB')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr.getvalue()


@pytest.fixture(scope="function")
def mock_model():
    """Мок ML модели для тестов"""

    class MockModel:
        def predict(self, x, verbose=0):
            # Возвращаем фиксированные предсказания
            batch_size = x.shape[0]
            num_labels = 15
            predictions = np.random.rand(batch_size, num_labels)
            # Делаем первые два лейбла более вероятными
            predictions[0, 0] = 0.8  # Cardiomegaly
            predictions[0, 1] = 0.7  # Effusion
            return predictions

    return MockModel()


@pytest.fixture(scope="function")
def mock_embedding_model():
    """Мок модели для эмбеддингов"""

    class MockEmbeddingModel:
        def predict(self, x, verbose=0):
            # Возвращаем фиксированный вектор эмбеддингов
            batch_size = x.shape[0]
            return np.random.rand(batch_size, 512)

    return MockEmbeddingModel()


@pytest.fixture(scope="session")
def qdrant_test_collection():
    """Имя тестовой коллекции Qdrant"""
    return "test_radiology_embeddings"


# Моковые данные для тестов
@pytest.fixture
def mock_prediction_response():
    """Мок ответа предсказания"""
    return {
        "detected": ["Cardiomegaly", "Effusion"],
        "predictions": [
            {"label": "Cardiomegaly", "probability": 0.8, "detected": True},
            {"label": "Effusion", "probability": 0.7, "detected": True},
            {"label": "Atelectasis", "probability": 0.3, "detected": False},
            {"label": "Mass", "probability": 0.2, "detected": False},
        ],
        "threshold": 0.5,
        "saved_to_db": False
    }


@pytest.fixture
def mock_user_data():
    """Мок данных пользователя"""
    return {
        "email": "test@example.com",
        "password": "testpassword123"
    }


@pytest.fixture(scope="session")
def test_files_dir(tmp_path_factory):
    """Временная директория для тестовых файлов"""
    return tmp_path_factory.mktemp("test_files")


# Хуки для настройки pytest
def pytest_configure(config):
    """Конфигурация pytest"""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "qdrant: mark test as requiring Qdrant"
    )

    # Устанавливаем переменные окружения для тестов
    os.environ["TESTING"] = "1"
    os.environ["DB_USER"] = "test"
    os.environ["DB_PASSWORD"] = "test"
    os.environ["DB_NAME"] = "test"
    os.environ["SECRET_KEY"] = "test_secret_key_min_32_characters_long"
    os.environ["ALGORITHM"] = "HS256"


def pytest_collection_modifyitems(config, items):
    """Модифицирует собранные тесты"""
    for item in items:
        # Добавляем маркер integration для тестов в папке integration
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Добавляем маркер slow для тестов с ML моделями
        if "test_predict" in item.nodeid or "test_model" in item.nodeid:
            item.add_marker(pytest.mark.slow)