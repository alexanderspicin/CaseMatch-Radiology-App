import io
import os
import subprocess
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime

from PIL import Image
from tensorflow.keras.models import load_model as load_keras_model
import numpy as np

if os.getenv('DOCKER_ENV'):
    BASE_DIR = Path('/app')
else:
    BASE_DIR = ''

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
        self.model_path = model_path
        self.model_version = None
        self.model_type = None
        self.model_params = None

    def pull_from_dvc(self) -> bool:
        try:
            os.chdir(str(BASE_DIR))

            # Проверяем наличие DVC
            if not (BASE_DIR / '.dvc').exists():
                return False

            result = subprocess.run(
                ['dvc', 'pull', 'models.dvc'],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                return True
            else:
                return False

        except Exception as e:
            return False

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
        except Exception as e:
            pass
        return None

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize(self.IMG_SIZE, Image.Resampling.LANCZOS)
        img_array = np.array(image, dtype='float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict_from_bytes(self, image_bytes: bytes, threshold: float = 0.5) -> Dict:
        """Предсказание из байтов изображения"""
        img = Image.open(io.BytesIO(image_bytes))
        return self.predict(img, threshold)

    def predict(self, image: Image.Image, threshold: float = 0.5) -> Dict:
        if self.model is None:
            raise ValueError('Model not loaded')
        img_array = self.preprocess_image(image)
        predictions = self.model.predict(img_array, verbose=0)[0]
        result = []
        for label, probability in zip(self.labels, predictions):
            result.append({'label': label, 'probability': float(probability),
                           'detected': True if probability >= threshold else False})
        result.sort(key=lambda x: x['probability'], reverse=True)
        detected = [r['label'] for r in result if r['detected']]
        return {
            'detected': detected,
            'predictions': result,
            'threshold': threshold,
        }


model_manager = ModelManager()
