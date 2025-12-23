import streamlit as st
import requests
import os
from PIL import Image
import io
import json
from datetime import datetime

# Конфигурация
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Настройка страницы
st.set_page_config(
    page_title="CaseMatch Radiology",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS стили
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        border: 2px solid #1f77b4;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .detected {
        background-color: #ffebee;
        color: #c62828;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Инициализация session state
if 'token' not in st.session_state:
    st.session_state.token = None
if 'user_data' not in st.session_state:
    st.session_state.user_data = None


def make_request(endpoint, method="GET", data=None, files=None, params=None, auth_required=True):
    """Выполняет запрос к API"""
    url = f"{API_URL}{endpoint}"
    headers = {}

    if auth_required and st.session_state.token:
        headers["Authorization"] = f"Bearer {st.session_state.token}"

    try:
        if method == "GET":
            response = requests.get(url, headers=headers, params=params)
        elif method == "POST":
            if files:
                # Для multipart/form-data не устанавливаем Content-Type
                response = requests.post(url, headers=headers, files=files, params=params)
            else:
                headers["Content-Type"] = "application/json"
                response = requests.post(url, headers=headers, json=data, params=params)

        return response
    except Exception as e:
        st.error(f"Ошибка соединения: {e}")
        return None


def login_page():
    """Страница входа"""
    st.markdown("<h1 class='main-header'>🏥 CaseMatch Radiology</h1>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Вход", "Регистрация"])

    with tab1:
        st.subheader("Вход в систему")
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Пароль", type="password", key="login_password")

        if st.button("Войти", use_container_width=True):
            if email and password:
                response = make_request(
                    "/user/login",
                    method="POST",
                    data={"email": email, "password": password},
                    auth_required=False
                )

                if response and response.status_code == 200:
                    st.session_state.token = response.json()
                    st.success("Успешный вход!")
                    st.rerun()
                else:
                    st.error("Неверный email или пароль")
            else:
                st.warning("Заполните все поля")

    with tab2:
        st.subheader("Регистрация")
        reg_email = st.text_input("Email", key="reg_email")
        reg_password = st.text_input("Пароль", type="password", key="reg_password")
        reg_password_confirm = st.text_input("Подтвердите пароль", type="password", key="reg_password_confirm")

        if st.button("Зарегистрироваться", use_container_width=True):
            if reg_email and reg_password and reg_password_confirm:
                if reg_password != reg_password_confirm:
                    st.error("Пароли не совпадают")
                else:
                    response = make_request(
                        "/user/register",
                        method="POST",
                        data={"email": reg_email, "password": reg_password},
                        auth_required=False
                    )

                    if response and response.status_code == 200:
                        st.success("Регистрация успешна! Теперь войдите в систему")
                    elif response:
                        st.error(f"Ошибка: {response.json().get('detail', 'Неизвестная ошибка')}")
            else:
                st.warning("Заполните все поля")


def get_user_info():
    """Получает информацию о пользователе"""
    response = make_request("/user/me", method="GET")
    if response and response.status_code == 200:
        st.session_state.user_data = response.json()
        return st.session_state.user_data
    return None


def sidebar_info():
    """Боковая панель с информацией о пользователе"""
    with st.sidebar:
        st.title("👤 Профиль")

        user_data = get_user_info()
        if user_data:
            st.write(f"**Email:** {user_data['email']}")
            st.write(f"**ID:** {str(user_data['id'])[:8]}...")

            st.divider()

            st.title("💰 Баланс")
            balance = user_data.get('balance', {})
            st.metric("Токены", f"{balance.get('amount', 0):.2f}")

            # Пополнение баланса
            st.subheader("Пополнить баланс")
            amount = st.number_input("Сумма (₽)", min_value=1.0, value=100.0, step=10.0)
            if st.button("Пополнить", use_container_width=True):
                response = make_request(
                    "/balance/credit",
                    method="GET",
                    params={"amount": amount}
                )
                if response and response.status_code == 200:
                    st.success(f"Баланс пополнен на {amount} ₽")
                    st.rerun()
                else:
                    st.error("Ошибка пополнения баланса")

            st.divider()

            if st.button("Выйти", use_container_width=True):
                st.session_state.token = None
                st.session_state.user_data = None
                st.rerun()


def prediction_page():
    """Страница предсказания"""
    st.markdown("<h1 class='main-header'>📊 Анализ рентгеновских снимков</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Загрузите снимок")
        uploaded_file = st.file_uploader(
            "Выберите изображение",
            type=["jpg", "jpeg", "png"],
            help="Поддерживаются форматы: JPG, PNG"
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Загруженное изображение", use_container_width=True)

            st.subheader("Параметры анализа")
            threshold = st.slider("Порог детекции", 0.0, 1.0, 0.5, 0.05)
            save_to_db = st.checkbox("Сохранить в базу данных", value=False,
                                     help="Сохранить снимок для поиска похожих случаев")

            if st.button("🔍 Анализировать", use_container_width=True, type="primary"):
                with st.spinner("Анализ изображения..."):
                    # Подготовка файла
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format='PNG')
                    img_byte_arr.seek(0)

                    files = {"image": ("image.png", img_byte_arr, "image/png")}
                    params = {
                        "threshold": threshold,
                        "save_to_db": save_to_db
                    }

                    response = make_request(
                        "/predict/predict",
                        method="POST",
                        files=files,
                        params=params
                    )

                    if response and response.status_code == 200:
                        result = response.json()
                        st.session_state.prediction_result = result
                    else:
                        st.error(f"Ошибка анализа: {response.text if response else 'Нет ответа'}")

    with col2:
        st.subheader("Результаты анализа")

        if 'prediction_result' in st.session_state and st.session_state.prediction_result:
            result = st.session_state.prediction_result

            # Фильтруем реальные патологии (исключаем "No Finding")
            detected = result.get('detected', [])
            diseases = [label for label in detected if label != 'No Finding']

            # Обнаруженные патологии
            if diseases:
                st.markdown("### 🚨 Обнаружены патологии:")
                for label in diseases:
                    st.markdown(f"<div class='detected'>• {label}</div>", unsafe_allow_html=True)
            else:
                st.success("✅ Патологий не обнаружено")

            st.divider()

            # Детальные результаты
            st.markdown("### 📋 Детальные результаты:")
            predictions = result.get('predictions', [])

            for pred in predictions[:5]:  # Показываем топ-5
                label = pred['label']
                prob = pred['probability']
                detected = pred['detected']
                is_disease = label != 'No Finding'

                col_a, col_b, col_c = st.columns([3, 1, 1])
                with col_a:
                    st.write(f"**{label}**")
                with col_b:
                    st.write(f"{prob * 100:.1f}%")
                with col_c:
                    # No Finding - зеленый если detected, болезни - красный если detected
                    if detected:
                        if is_disease:
                            st.write("🔴")  # Болезнь обнаружена - плохо
                        else:
                            st.write("🟢")  # No Finding - хорошо
                    else:
                        if is_disease:
                            st.write("🟢")  # Болезнь не обнаружена - хорошо
                        else:
                            st.write("⚪")  # No Finding не обнаружен - нейтрально

                st.progress(prob)

            # Информация о сохранении
            if result.get('saved_to_db'):
                point_id = result.get('point_id', 'N/A')
                display_id = point_id[:8] if point_id and point_id != 'N/A' else 'N/A'
                st.success(f"✅ Снимок сохранен в базу данных (ID: {display_id}...)")
        else:
            st.info("Загрузите изображение и нажмите 'Анализировать' для получения результатов")


def search_similar_page():
    """Страница поиска похожих случаев"""
    st.markdown("<h1 class='main-header'>🔎 Поиск похожих случаев</h1>", unsafe_allow_html=True)

    st.write("Загрузите снимок для поиска похожих случаев в базе данных")

    uploaded_file = st.file_uploader(
        "Выберите изображение для поиска",
        type=["jpg", "jpeg", "png"],
        key="search_file"
    )

    col1, col2 = st.columns(2)
    with col1:
        limit = st.slider("Количество результатов", 1, 20, 5)
    with col2:
        score_threshold = st.slider("Порог схожести", 0.0, 1.0, 0.7, 0.05)

    if uploaded_file and st.button("Искать", use_container_width=True, type="primary"):
        with st.spinner("Поиск похожих случаев..."):
            image = Image.open(uploaded_file)

            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)

            files = {"image": ("image.png", img_byte_arr, "image/png")}
            params = {
                "limit": limit,
                "score_threshold": score_threshold
            }

            response = make_request(
                "/predict/search-similar",
                method="POST",
                files=files,
                params=params
            )

            if response and response.status_code == 200:
                results = response.json()

                if results['count'] > 0:
                    st.success(f"Найдено {results['count']} похожих случаев")

                    for idx, result in enumerate(results['results'], 1):
                        with st.expander(f"Случай #{idx} - Схожесть: {result['score'] * 100:.1f}%"):
                            payload = result['payload']

                            col_a, col_b = st.columns([1, 2])

                            with col_a:
                                result_id = str(result['id'])[:8] if result.get('id') else 'N/A'
                                st.write(f"**ID:** {result_id}...")
                                timestamp = payload.get('timestamp', 'N/A')
                                display_date = timestamp[:10] if timestamp and timestamp != 'N/A' else 'N/A'
                                st.write(f"**Дата:** {display_date}")
                                st.write(f"**Пользователь:** {payload.get('user_email', 'N/A')}")

                            with col_b:
                                st.write("**Обнаружено:**")
                                detected = payload.get('detected_labels', [])
                                # Фильтруем реальные патологии
                                diseases = [label for label in detected if label != 'No Finding']
                                if diseases:
                                    for label in diseases:
                                        st.markdown(f"• {label}")
                                else:
                                    st.write("Патологий не обнаружено")
                else:
                    st.warning("Похожих случаев не найдено")
            else:
                error_msg = response.text if response else 'Нет ответа'
                st.error(f"Ошибка поиска: {error_msg}")


def main_app():
    """Главное приложение"""
    sidebar_info()

    # Навигация
    page = st.sidebar.radio(
        "Навигация",
        ["📊 Анализ снимков", "🔎 Поиск похожих случаев", "📈 История транзакций"]
    )

    if page == "📊 Анализ снимков":
        prediction_page()
    elif page == "🔎 Поиск похожих случаев":
        search_similar_page()
    elif page == "📈 История транзакций":
        st.markdown("<h1 class='main-header'>📈 История транзакций</h1>", unsafe_allow_html=True)

        user_data = st.session_state.user_data
        if user_data:
            transactions = user_data.get('transactions', [])

            if transactions:
                for trans in transactions:
                    with st.expander(f"{trans['transaction_type']} - {trans['amount']} токенов"):
                        st.write(f"**ID:** {trans['id']}")
                        st.write(f"**Дата:** {trans['timestamp']}")
                        st.write(f"**Статус:** {trans['transaction_status']}")
            else:
                st.info("Транзакций пока нет")


def main():
    """Основная функция приложения"""
    if st.session_state.token is None:
        login_page()
    else:
        main_app()


if __name__ == "__main__":
    main()