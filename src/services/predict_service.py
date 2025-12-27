import io
import logging
import os
import subprocess
import base64
import uuid as uuid_lib
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime

from PIL import Image
from tensorflow.keras.models import load_model as load_keras_model
from tensorflow.keras import Model
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

if os.getenv('DOCKER_ENV'):
    BASE_DIR = Path('/app')
else:
    BASE_DIR = Path('.')

MODELS_DIR = BASE_DIR / 'models'
MODEL_PATH = MODELS_DIR / 'advanced_v3_all_in_one_smooth.h5'


class ModelManager:

    def __init__(self, model_path: Path = MODEL_PATH):
        self.labels = [
            'Atelectasis', 'Cardiomegaly', 'Effusion',
            'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
            'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
            'Pleural_Thickening', 'Hernia', 'No Finding'
        ]
        self.IMG_SIZE = (480, 480)
        self.model = None
        self.embedding_model = None
        self.model_path = model_path
        self.model_version = None
        self.model_type = None
        self.model_params = None

        # Qdrant клиент
        qdrant_host = os.getenv('QDRANT_HOST', 'qdrant')
        qdrant_port = int(os.getenv('QDRANT_PORT', 6333))
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = "radiology_embeddings"

    def pull_from_dvc(self) -> bool:
        try:
            os.chdir(str(BASE_DIR))
            if not (BASE_DIR / '.dvc').exists():
                return False

            result = subprocess.run(
                ['dvc', 'pull', 'models.dvc'],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except Exception:
            return False

    def _create_embedding_model(self):
        """Создает модель для извлечения эмбеддингов"""
        if self.model is None:
            print("WARNING: Cannot create embedding model - main model is None", flush=True)
            return None

        try:
            embedding_layer = None

            # Ищем подходящий слой по имени и размерности
            for layer in self.model.layers:
                layer_name = layer.name.lower()
                layer_type = type(layer).__name__

                # Пропускаем вложенные Functional модели (как efficientnetv2-s)
                if layer_type == 'Functional':
                    print(f"DEBUG: Skipping Functional layer: {layer.name}", flush=True)
                    continue

                # Пропускаем Input слои
                if 'input' in layer_name:
                    continue

                try:
                    output_shape = layer.output.shape
                    print(f"DEBUG: Layer {layer.name}, type: {layer_type}, shape: {output_shape}", flush=True)

                    # Ищем Dropout или Dense с большой размерностью (1280 или 256)
                    if len(output_shape) == 2 and output_shape[-1]:
                        dim = output_shape[-1]
                        # Берём первый слой с размерностью >= 256 (но не финальный с 15)
                        if dim >= 256 and dim != 15:
                            embedding_layer = layer
                            print(f"DEBUG: Selected layer: {layer.name}, dim: {dim}", flush=True)
                            break
                except Exception as e:
                    print(f"DEBUG: Could not get shape for {layer.name}: {e}", flush=True)
                    continue

            if embedding_layer is None:
                print("ERROR: Could not find suitable embedding layer", flush=True)
                return None

            print(f"DEBUG: Using layer for embeddings: {embedding_layer.name}", flush=True)

            self.embedding_model = Model(
                inputs=self.model.input,
                outputs=embedding_layer.output
            )

            print(f"DEBUG: Embedding model created, output shape: {self.embedding_model.output.shape}", flush=True)

        except Exception as e:
            print(f"ERROR: Failed to create embedding model: {e}", flush=True)
            import traceback
            traceback.print_exc()
            self.embedding_model = None

    def _ensure_qdrant_collection(self):
        """Создает коллекцию в Qdrant если её нет"""
        try:
            collections = self.qdrant_client.get_collections().collections
            collection_exists = any(c.name == self.collection_name for c in collections)

            if not collection_exists:
                # Определяем размер вектора эмбеддингов
                dummy_img = np.zeros((1, *self.IMG_SIZE, 3), dtype='float32')
                if self.embedding_model:
                    dummy_embedding = self.embedding_model.predict(dummy_img, verbose=0)
                    vector_size = int(np.prod(dummy_embedding.shape[1:]))
                else:
                    vector_size = 512  # Default

                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                )
        except Exception as e:
            pass

    def load_model(self) -> bool:
        try:
            if not self.model_path.exists():
                if self.pull_from_dvc():
                    if not self.model_path.exists():
                        return False
                else:
                    return False

            self.model = load_keras_model(self.model_path, compile=False)
            model_name = type(self.model).__name__
            self.model_type = model_name

            if hasattr(self.model, 'get_params'):
                self.model_params = self.model.get_params()

            stat = self.model_path.stat()
            self.model_version = f"v{datetime.fromtimestamp(stat.st_mtime).strftime('%Y%m%d_%H%M%S')}"

            # Создаем модель для эмбеддингов
            self._create_embedding_model()

            # Инициализируем Qdrant коллекцию
            self._ensure_qdrant_collection()

            return True

        except Exception as e:
            return False

    def get_dvc_remote(self) -> Optional[str]:
        try:
            result = subprocess.run(
                ['dvc', 'remote', 'list'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        if image.mode != 'L':
            image = image.convert('L')

        image = image.resize(self.IMG_SIZE, Image.Resampling.LANCZOS)

        img_array = np.array(image, dtype='float32')
        img_array = np.stack([img_array, img_array, img_array], axis=-1)
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    def _extract_embeddings(self, img_array: np.ndarray) -> np.ndarray:
        """Извлекает эмбеддинги из изображения"""
        if self.embedding_model is None:
            raise ValueError("Embedding model not initialized")

        embeddings = self.embedding_model.predict(img_array, verbose=0)
        # Преобразуем в 1D вектор
        embeddings_flat = embeddings.flatten()
        return embeddings_flat

    def _image_to_base64(self, image: Image.Image) -> str:
        """Конвертирует изображение в base64"""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    def _save_to_qdrant(
            self,
            embeddings: np.ndarray,
            image: Image.Image,
            predictions: Dict,
            user_id: Optional[str] = None,
            metadata: Optional[Dict] = None
    ) -> str:
        """Сохраняет эмбеддинги и изображение в Qdrant"""
        try:
            point_id = str(uuid_lib.uuid4())

            payload = {
                "image_base64": self._image_to_base64(image),
                "predictions": predictions,
                "timestamp": datetime.now().isoformat(),
                "user_id": user_id,
                "detected_labels": predictions.get("detected", []),
                "threshold": predictions.get("threshold", 0.5),
            }

            if metadata:
                payload.update(metadata)

            point = PointStruct(
                id=point_id,
                vector=embeddings.tolist(),
                payload=payload
            )

            # Работает в обеих версиях
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )

            print(f"DEBUG: Saved to Qdrant with id: {point_id}", flush=True)
            return point_id
        except Exception as e:
            print(f"ERROR: Failed to save to Qdrant: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return None

    def predict_from_bytes(
            self,
            image_bytes: bytes,
            threshold: float = 0.5,
            save_to_db: bool = False,
            user_id: Optional[str] = None,
            metadata: Optional[Dict] = None
    ) -> Dict:
        """Предсказание из байтов изображения"""
        img = Image.open(io.BytesIO(image_bytes))
        return self.predict(img, threshold, save_to_db, user_id, metadata)

    def predict(
            self,
            image: Image.Image,
            threshold: float = 0.5,
            save_to_db: bool = False,
            user_id: Optional[str] = None,
            metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Выполняет предсказание и опционально сохраняет в Qdrant

        Args:
            image: PIL изображение
            threshold: Порог для детекции
            save_to_db: Сохранять ли пример в Qdrant
            user_id: ID пользователя
            metadata: Дополнительные метаданные

        Returns:
            Dict с результатами предсказания и point_id если сохранено
        """
        if self.model is None:
            raise ValueError('Model not loaded')

        img_array = self.preprocess_image(image)
        predictions = self.model.predict(img_array, verbose=0)[0]

        result = []
        for label, probability in zip(self.labels, predictions):
            result.append({
                'label': label,
                'probability': float(probability),
                'detected': True if probability >= threshold else False
            })

        result.sort(key=lambda x: x['probability'], reverse=True)
        detected = [r['label'] for r in result if r['detected']]

        prediction_result = {
            'detected': detected,
            'predictions': result,
            'threshold': threshold,
        }

        # Сохраняем в Qdrant если требуется
        if save_to_db:
            try:
                embeddings = self._extract_embeddings(img_array)
                point_id = self._save_to_qdrant(
                    embeddings=embeddings,
                    image=image,
                    predictions=prediction_result,
                    user_id=user_id,
                    metadata=metadata
                )
                prediction_result['point_id'] = point_id
                prediction_result['saved_to_db'] = True
            except Exception as e:
                print('Failed to save to Qdrant', flush=True)
                print(e, flush=True)
                prediction_result['saved_to_db'] = False
                prediction_result['error'] = str(e)

        return prediction_result

    def search_similar(
            self,
            image: Image.Image,
            limit: int = 5,
            score_threshold: float = 0.7
    ) -> list:
        """Ищет похожие изображения в Qdrant"""

        # Ленивая инициализация embedding model
        if self.embedding_model is None and self.model is not None:
            print("DEBUG: Lazy initialization of embedding model", flush=True)
            self._create_embedding_model()

        if self.embedding_model is None:
            raise ValueError("Embedding model not initialized")

        img_array = self.preprocess_image(image)
        embeddings = self._extract_embeddings(img_array)

        search_results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=embeddings.tolist(),
            limit=limit,
            score_threshold=score_threshold
        )

        return [
            {
                "id": point.id,
                "score": point.score,
                "payload": point.payload
            }
            for point in search_results.points
        ]


model_manager = ModelManager()