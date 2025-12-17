import os
import subprocess
import joblib
from pathlib import Path
from typing import Optional
from datetime import datetime
from tensorflow.keras.models import load_model as load_keras_model

if os.getenv('DOCKER_ENV'):
    BASE_DIR = Path('/app')
else:
    BASE_DIR = ''

MODELS_DIR = BASE_DIR / 'models'

MODEL_PATH = MODELS_DIR / 'advanced_v3_all_in_one_smooth.h5'



class ModelManager:

    def __init__(self, model_path: Path = MODEL_PATH):
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
            print(e)
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
            print(f"Error loading model: {e}")
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


model_manager = ModelManager()

