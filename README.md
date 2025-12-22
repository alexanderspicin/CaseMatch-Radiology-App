# CaseMatch Radiology App 🏥

Масштабируемое приложение для анализа рентгеновских снимков с использованием ML и векторного поиска похожих случаев.

## 🚀 Возможности

- **ML Анализ**: Автоматическое определение патологий на рентгеновских снимках
- **Векторный поиск**: Поиск похожих случаев с использованием Qdrant
- **Streamlit UI**: Удобный веб-интерфейс для взаимодействия
- **Масштабируемость**: Горизонтальное масштабирование через Docker Compose
- **Система балансов**: Управление токенами и транзакциями пользователей
- **JWT Authentication**: Безопасная аутентификация пользователей

## 📋 Технологический стек

### Backend
- **FastAPI** - современный веб-фреймворк
- **PostgreSQL** - реляционная база данных
- **Qdrant** - векторная база данных для поиска похожих случаев
- **Redis** - кеширование и очереди задач
- **TensorFlow/Keras** - ML модель для анализа снимков

### Frontend
- **Streamlit** - интерактивный веб-интерфейс

### Инфраструктура
- **Docker & Docker Compose** - контейнеризация
- **Nginx** - load balancer и reverse proxy
- **Alembic** - миграции базы данных

## 🏗️ Архитектура

```
┌─────────────┐
│   Streamlit │ ───┐
└─────────────┘    │
                   ├──► ┌───────┐      ┌──────────┐
┌─────────────┐    │    │ Nginx │ ───► │ API (x3) │
│   Клиент    │ ───┘    └───────┘      └──────────┘
└─────────────┘                             │
                                            ├──► PostgreSQL
                                            ├──► Qdrant
                                            └──► Redis
```

## 📦 Установка и запуск

### Предварительные требования

- Docker >= 20.10
- Docker Compose >= 2.0
- NVIDIA Docker (опционально, для GPU)

### Быстрый старт

1. **Клонируйте репозиторий**
```bash
git clone https://github.com/alexanderspicin/CaseMatch-Radiology-App.git
cd CaseMatch-Radiology-App
```

2. **Создайте .env файл**
```bash
cp .env.example .env
# Отредактируйте .env и добавьте свои данные
```

3. **Инициализация проекта (первый запуск)**
```bash
docker-compose build
docker-compose up -d db qdrant redis
docker-compose up -d api
docker-compose exec api alembic upgrade head
docker-compose up -d
```

4. **Доступ к сервисам**
- API: http://localhost
- Streamlit: http://localhost:8501
- PgAdmin: http://localhost:5050
- Qdrant Dashboard: http://localhost:6333/dashboard

## 🔧 Конфигурация

### Файл .env

```env
# Database
DB_USER=postgres
DB_PASSWORD=your_secure_password
DB_NAME=casematch
DB_HOST=db
DB_PORT=5432

# Security
SECRET_KEY=your_secret_key_here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Qdrant
QDRANT_HOST=qdrant
QDRANT_PORT=6333

# Redis
REDIS_URL=redis://redis:6379/0

# PgAdmin
PGADMIN_EMAIL=admin@admin.com
PGADMIN_PASSWORD=admin

# Docker
DOCKER_ENV=true
```

## 📝 API Документация

После запуска API документация доступна по адресу:
- Swagger UI: http://localhost/docs
- ReDoc: http://localhost/redoc

### Основные endpoints

#### Аутентификация
- `POST /user/register` - Регистрация пользователя
- `POST /user/login` - Вход в систему
- `GET /user/me` - Получить данные текущего пользователя

#### Предсказания
- `POST /predict/predict` - Анализ рентгеновского снимка
  - Параметры:
    - `image`: Файл изображения (JPEG/PNG)
    - `threshold`: Порог детекции (0.0-1.0)
    - `save_to_db`: Сохранить в Qdrant (bool)
- `POST /predict/search-similar` - Поиск похожих случаев
- `GET /predict/health` - Статус ML модели

#### Баланс
- `GET /balance/credit?amount=100` - Пополнить баланс

## 🔍 Использование Qdrant

### Сохранение примеров

При вызове `/predict/predict` с параметром `save_to_db=true`:
1. ML модель извлекает эмбеддинги с предпоследнего слоя
2. Эмбеддинги и метаданные сохраняются в Qdrant
3. Изображение кодируется в base64 и сохраняется в payload

### Поиск похожих случаев

```python
# Через API
response = requests.post(
    "http://localhost/predict/search-similar",
    files={"image": open("xray.jpg", "rb")},
    data={"limit": 5, "score_threshold": 0.7},
    headers={"Authorization": f"Bearer {token}"}
)
```

## 📊 Масштабирование

### Горизонтальное масштабирование

```bash
# Масштабирование API
docker-compose up -d --scale api=5

# Проверка
docker-compose ps
```

### Вертикальное масштабирование

Отредактируйте `docker-compose.yml`:

```yaml
api:
  deploy:
    resources:
      limits:
        cpus: '4'      # Увеличьте CPU
        memory: 8G     # Увеличьте память
```

## 🔐 Безопасность

- JWT токены для аутентификации
- Хеширование паролей с Argon2
- Rate limiting в Nginx
- CORS защита
- CSRF защита в Streamlit
- Изолированные Docker сети

## 📚 Структура проекта

```
CaseMatch-Radiology-App/
├── src/
│   ├── api/
│   │   ├── main.py           # Главный файл FastAPI
│   │   ├── predict.py        # Endpoints для предсказаний
│   │   ├── users.py          # Endpoints для пользователей
│   │   └── balance.py        # Endpoints для баланса
│   ├── models/               # SQLAlchemy модели
│   ├── schemas/              # Pydantic схемы
|   ├── streamlit_app/
│   │   ├── streamlit_app.py
│   ├── services/             # Бизнес логика
│   │   ├── predict_service.py  # ML сервис + Qdrant
│   │   ├── auth_service.py     # Аутентификация
│   │   └── ...
│   ├── core/                 # Конфигурация
│   └── db/                   # База данных
├── alembic/                  # Миграции БД
├── models/                   # ML модели
├── docker-compose.yml        # Development compose
├── docker-compose.prod.yml   # Production compose
├── nginx.conf                # Nginx конфигурация
├── Dockerfile                # API Dockerfile
├── Dockerfile.streamlit      # Streamlit Dockerfile
├── requirements.txt          # Python зависимости
└── README.md                 # Этот файл
```

## 🐛 Troubleshooting

### Проблемы с GPU

Если у вас нет NVIDIA GPU, удалите из `docker-compose.yml`:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
```

### Ошибки миграций

```bash
# Сброс миграций
docker-compose exec api alembic downgrade base
docker-compose exec api alembic upgrade head
```

### Проблемы с Qdrant

```bash
# Пересоздать коллекцию
docker-compose exec qdrant /bin/sh
# Внутри контейнера можно использовать Qdrant API
```

## 🤝 Вклад в проект

1. Fork проекта
2. Создайте feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit изменений (`git commit -m 'Add some AmazingFeature'`)
4. Push в branch (`git push origin feature/AmazingFeature`)
5. Откройте Pull Request

## 📄 Лицензия

Этот проект распространяется под лицензией MIT.

## 👨‍💻 Автор

Alexander Spicin

## 🙏 Благодарности

- FastAPI за отличный фреймворк
- Qdrant за векторную базу данных
- Streamlit за простой UI
- Сообщество open-source

---

**Примечание**: Этот проект предназначен для образовательных целей. Для использования в медицинских целях требуется соответствующая сертификация и валидация.