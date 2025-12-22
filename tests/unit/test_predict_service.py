import pytest
import numpy as np
from PIL import Image
import io
from unittest.mock import patch, MagicMock, Mock
from pathlib import Path


class TestModelManager:
    """Тесты для ModelManager"""

    def test_model_manager_initialization(self):
        """Тест инициализации ModelManager"""
        from src.services.predict_service import ModelManager

        manager = ModelManager()

        assert manager.labels is not None
        assert len(manager.labels) == 15
        assert manager.IMG_SIZE == (480, 480)
        assert manager.model is None  # До загрузки модели

    def test_preprocess_image_rgb(self):
        """Тест предобработки RGB изображения"""
        from src.services.predict_service import ModelManager

        manager = ModelManager()
        img = Image.new('RGB', (100, 100), color='red')

        processed = manager.preprocess_image(img)

        assert processed.shape == (1, 480, 480, 3)
        assert processed.dtype == np.float32
        assert 0 <= processed.min() <= 1
        assert 0 <= processed.max() <= 1

    def test_preprocess_image_grayscale(self):
        """Тест предобработки grayscale изображения"""
        from src.services.predict_service import ModelManager

        manager = ModelManager()
        img = Image.new('L', (100, 100), color=128)  # Grayscale

        processed = manager.preprocess_image(img)

        # Должно быть конвертировано в RGB
        assert processed.shape == (1, 480, 480, 3)
        assert processed.dtype == np.float32

    def test_preprocess_image_rgba(self):
        """Тест предобработки RGBA изображения"""
        from src.services.predict_service import ModelManager

        manager = ModelManager()
        img = Image.new('RGBA', (100, 100), color=(255, 0, 0, 255))

        processed = manager.preprocess_image(img)

        # Должно быть конвертировано в RGB
        assert processed.shape == (1, 480, 480, 3)

    def test_preprocess_image_normalization(self):
        """Тест нормализации пикселей"""
        from src.services.predict_service import ModelManager

        manager = ModelManager()
        # Создаем изображение с максимальными значениями
        img_array = np.full((100, 100, 3), 255, dtype=np.uint8)
        img = Image.fromarray(img_array)

        processed = manager.preprocess_image(img)

        # Проверяем что значения нормализованы к [0, 1]
        assert processed.max() <= 1.0
        assert processed.min() >= 0.0

    @patch('src.services.predict_service.ModelManager._extract_embeddings')
    @patch('src.services.predict_service.ModelManager._save_to_qdrant')
    def test_predict_without_save(self, mock_save, mock_extract, mock_model):
        """Тест предсказания без сохранения"""
        from src.services.predict_service import ModelManager

        manager = ModelManager()
        manager.model = mock_model

        img = Image.new('RGB', (480, 480), color='blue')
        result = manager.predict(img, threshold=0.5, save_to_db=False)

        assert "detected" in result
        assert "predictions" in result
        assert "threshold" in result
        assert result["threshold"] == 0.5

        # Не должно было вызваться сохранение
        mock_save.assert_not_called()
        mock_extract.assert_not_called()

    @patch('src.services.predict_service.ModelManager._extract_embeddings')
    @patch('src.services.predict_service.ModelManager._save_to_qdrant')
    def test_predict_with_save(self, mock_save, mock_extract, mock_model):
        """Тест предсказания с сохранением"""
        from src.services.predict_service import ModelManager

        manager = ModelManager()
        manager.model = mock_model
        manager.embedding_model = Mock()

        mock_extract.return_value = np.random.rand(512)
        mock_save.return_value = "test-point-id"

        img = Image.new('RGB', (480, 480), color='blue')
        result = manager.predict(
            img,
            threshold=0.5,
            save_to_db=True,
            user_id="user-123"
        )

        assert result.get("saved_to_db") is True
        assert "point_id" in result

        # Должно было вызваться сохранение
        mock_extract.assert_called_once()
        mock_save.assert_called_once()

    def test_image_to_base64(self):
        """Тест конвертации изображения в base64"""
        from src.services.predict_service import ModelManager

        manager = ModelManager()
        img = Image.new('RGB', (100, 100), color='green')

        base64_str = manager._image_to_base64(img)

        assert isinstance(base64_str, str)
        assert len(base64_str) > 0

        # Проверяем что можно декодировать обратно
        import base64
        decoded = base64.b64decode(base64_str)
        assert len(decoded) > 0

    def test_predict_threshold_filtering(self, mock_model):
        """Тест фильтрации по порогу"""
        from src.services.predict_service import ModelManager

        manager = ModelManager()
        manager.model = mock_model

        img = Image.new('RGB', (480, 480))
        result = manager.predict(img, threshold=0.7)

        # Проверяем что detected содержит только предсказания >= 0.7
        for pred in result["predictions"]:
            if pred["detected"]:
                assert pred["probability"] >= 0.7
            else:
                assert pred["probability"] < 0.7

    def test_predictions_sorted(self, mock_model):
        """Тест что предсказания отсортированы по вероятности"""
        from src.services.predict_service import ModelManager

        manager = ModelManager()
        manager.model = mock_model

        img = Image.new('RGB', (480, 480))
        result = manager.predict(img, threshold=0.5)

        predictions = result["predictions"]
        probabilities = [p["probability"] for p in predictions]

        # Проверяем что список отсортирован в убывающем порядке
        assert probabilities == sorted(probabilities, reverse=True)

    def test_predict_from_bytes(self, mock_model):
        """Тест предсказания из байтов"""
        from src.services.predict_service import ModelManager

        manager = ModelManager()
        manager.model = mock_model

        img = Image.new('RGB', (480, 480), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        result = manager.predict_from_bytes(
            img_bytes.getvalue(),
            threshold=0.5
        )

        assert "detected" in result
        assert "predictions" in result


class TestQdrantIntegration:
    """Тесты интеграции с Qdrant"""

    @pytest.mark.qdrant
    @patch('src.services.predict_service.QdrantClient')
    def test_ensure_collection_creates_collection(self, mock_client):
        """Тест создания коллекции в Qdrant"""
        from src.services.predict_service import ModelManager

        # Мокаем что коллекции не существует
        mock_client_instance = mock_client.return_value
        mock_client_instance.get_collections.return_value.collections = []

        manager = ModelManager()
        manager.qdrant_client = mock_client_instance
        manager._ensure_qdrant_collection()

        # Проверяем что create_collection был вызван
        mock_client_instance.create_collection.assert_called_once()

    @pytest.mark.qdrant
    @patch('src.services.predict_service.QdrantClient')
    def test_ensure_collection_existing(self, mock_client):
        """Тест когда коллекция уже существует"""
        from src.services.predict_service import ModelManager
        from qdrant_client.models import CollectionInfo

        # Мокаем что коллекция существует
        mock_client_instance = mock_client.return_value
        mock_collection = Mock()
        mock_collection.name = "radiology_embeddings"
        mock_client_instance.get_collections.return_value.collections = [mock_collection]

        manager = ModelManager()
        manager.qdrant_client = mock_client_instance
        manager._ensure_qdrant_collection()

        # Не должно было вызваться создание
        mock_client_instance.create_collection.assert_not_called()

    @pytest.mark.qdrant
    @patch('src.services.predict_service.QdrantClient')
    def test_save_to_qdrant(self, mock_client, mock_model):
        """Тест сохранения в Qdrant"""
        from src.services.predict_service import ModelManager

        mock_client_instance = mock_client.return_value

        manager = ModelManager()
        manager.qdrant_client = mock_client_instance
        manager.model = mock_model

        embeddings = np.random.rand(512)
        img = Image.new('RGB', (100, 100))
        predictions = {
            "detected": ["Cardiomegaly"],
            "predictions": [],
            "threshold": 0.5
        }

        point_id = manager._save_to_qdrant(
            embeddings=embeddings,
            image=img,
            predictions=predictions,
            user_id="user-123"
        )

        assert point_id is not None
        mock_client_instance.upsert.assert_called_once()

    @pytest.mark.qdrant
    @patch('src.services.predict_service.QdrantClient')
    def test_search_similar(self, mock_client, mock_model, mock_embedding_model):
        """Тест поиска похожих случаев"""
        from src.services.predict_service import ModelManager

        mock_client_instance = mock_client.return_value

        # Мокаем результаты поиска
        mock_result = Mock()
        mock_result.id = "result-id-1"
        mock_result.score = 0.95
        mock_result.payload = {
            "detected_labels": ["Cardiomegaly"],
            "timestamp": "2024-01-01"
        }
        mock_client_instance.search.return_value = [mock_result]

        manager = ModelManager()
        manager.qdrant_client = mock_client_instance
        manager.embedding_model = mock_embedding_model

        img = Image.new('RGB', (480, 480))
        results = manager.search_similar(img, limit=5, score_threshold=0.7)

        assert len(results) == 1
        assert results[0]["id"] == "result-id-1"
        assert results[0]["score"] == 0.95

        mock_client_instance.search.assert_called_once()


class TestEmbeddingExtraction:
    """Тесты извлечения эмбеддингов"""

    @patch('src.services.predict_service.Model')
    def test_extract_embeddings_shape(self, mock_keras_model, mock_embedding_model):
        """Тест формы эмбеддингов"""
        from src.services.predict_service import ModelManager

        manager = ModelManager()
        manager.embedding_model = mock_embedding_model

        img_array = np.random.rand(1, 480, 480, 3).astype(np.float32)
        embeddings = manager._extract_embeddings(img_array)

        # Проверяем что эмбеддинги одномерные
        assert len(embeddings.shape) == 1
        assert embeddings.shape[0] > 0

    def test_embeddings_consistency(self, mock_embedding_model):
        """Тест что одинаковые изображения дают одинаковые эмбеддинги"""
        from src.services.predict_service import ModelManager

        # Делаем мок детерминированным
        def deterministic_predict(x, verbose=0):
            # Возвращаем одинаковые эмбеддинги для одинаковых входов
            return np.ones((x.shape[0], 512)) * x.mean()

        mock_embedding_model.predict = deterministic_predict

        manager = ModelManager()
        manager.embedding_model = mock_embedding_model

        img = Image.new('RGB', (480, 480), color='blue')
        img_array1 = manager.preprocess_image(img)
        img_array2 = manager.preprocess_image(img)

        emb1 = manager._extract_embeddings(img_array1)
        emb2 = manager._extract_embeddings(img_array2)

        np.testing.assert_array_almost_equal(emb1, emb2)


class TestModelLoading:
    """Тесты загрузки модели"""

    @patch('src.services.predict_service.load_keras_model')
    @patch('src.services.predict_service.Path.exists')
    def test_load_model_success(self, mock_exists, mock_load):
        """Тест успешной загрузки модели"""
        from src.services.predict_service import ModelManager

        mock_exists.return_value = True
        mock_model = Mock()
        mock_load.return_value = mock_model

        manager = ModelManager()
        result = manager.load_model()

        assert result is True
        assert manager.model is not None

    @patch('src.services.predict_service.Path.exists')
    def test_load_model_file_not_found(self, mock_exists):
        """Тест когда файл модели не найден"""
        from src.services.predict_service import ModelManager

        mock_exists.return_value = False

        manager = ModelManager()
        result = manager.load_model()

        assert result is False
        assert manager.model is None

    @patch('src.services.predict_service.load_keras_model')
    @patch('src.services.predict_service.Path.exists')
    def test_load_model_creates_embedding_model(self, mock_exists, mock_load):
        """Тест что создается модель для эмбеддингов"""
        from src.services.predict_service import ModelManager

        mock_exists.return_value = True
        mock_model = Mock()
        mock_model.layers = [Mock(), Mock()]
        mock_load.return_value = mock_model

        manager = ModelManager()

        with patch.object(manager, '_create_embedding_model') as mock_create:
            manager.load_model()
            mock_create.assert_called_once()


class TestPredictionLabels:
    """Тесты для меток предсказаний"""

    def test_all_labels_present(self):
        """Тест что все 15 меток присутствуют"""
        from src.services.predict_service import ModelManager

        manager = ModelManager()
        expected_labels = [
            'Atelectasis', 'Cardiomegaly', 'Effusion',
            'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
            'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
            'Pleural_Thickening', 'Hernia', 'No Finding'
        ]

        assert len(manager.labels) == 15
        for label in expected_labels:
            assert label in manager.labels

    def test_predictions_have_all_labels(self, mock_model):
        """Тест что предсказания содержат все метки"""
        from src.services.predict_service import ModelManager

        manager = ModelManager()
        manager.model = mock_model

        img = Image.new('RGB', (480, 480))
        result = manager.predict(img, threshold=0.5)

        prediction_labels = [p["label"] for p in result["predictions"]]

        assert len(prediction_labels) == 15
        assert set(prediction_labels) == set(manager.labels)