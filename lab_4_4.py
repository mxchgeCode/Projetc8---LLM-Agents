"""Модуль для демонстрации Function Calling.

Содержит два варианта реализации агента-помощника магазина электроники:
- Вариант A: smolagents CodeAgent с инструментом поиска по каталогу
- Вариант B: Structured Output (Pydantic + OpenAI SDK)
"""

import json
import logging
import os
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, Field


# ── Данные (каталог товаров) ───────────────────────────────────────────

CATALOG = {
    "iPhone 15": "Смартфон Apple, 128 ГБ, камера 48 Мп, цена 89 990 руб.",
    "Samsung Galaxy S24": "Смартфон Samsung, 256 ГБ, AI-функции, цена 74 990 руб.",
    "MacBook Air M3": "Ноутбук Apple, чип M3, 16 ГБ RAM, цена 124 990 руб.",
    "Lenovo ThinkPad": "Ноутбук для работы, 16 ГБ RAM, цена 67 990 руб. Нет в наличии.",
    "AirPods Pro 2": "Наушники Apple с шумоподавлением, цена 19 990 руб.",
    "Sony WH-1000XM5": "Полноразмерные наушники Sony, цена 24 990 руб.",
    "iPad Air": "Планшет Apple, чип M2, 10.9 дюймов, цена 59 990 руб.",
    "Samsung Tab S9": "Планшет Samsung, AMOLED, цена 49 990 руб. Нет в наличии.",
    "Apple Watch SE": "Умные часы Apple, цена 29 990 руб.",
    "Xiaomi Band 8": "Фитнес-браслет Xiaomi, цена 3 990 руб.",
}


# ── Загрузка конфигурации ──────────────────────────────────────────────

def load_env_file(file_path: str = ".env") -> None:
    """Загружает переменные окружения из файла .env."""
    env_path = Path(file_path)
    if not env_path.exists():
        raise FileNotFoundError(
            f"Файл {file_path} не найден. Создайте его с нужными переменными."
        )

    with env_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition("=")
            if key and value:
                os.environ[key.strip()] = value.strip()


def get_api_config() -> tuple[str, str, str]:
    """Получает конфигурацию API из переменных окружения."""
    api_key = os.environ.get("API_KEY")
    base_url = os.environ.get("BASE_URL")
    model_name = os.environ.get("MODEL_NAME")

    if not all((api_key, base_url, model_name)):
        raise ValueError(
            "Не все обязательные переменные окружения заданы. "
            "Проверьте файл .env"
        )

    if not base_url.startswith(("http://", "https://")):
        raise ValueError(
            "BASE_URL должен начинаться с 'http://' или 'https://'. "
            f"Текущее значение: {base_url}"
        )

    return api_key, base_url, model_name


# ── Вариант A: smolagents ──────────────────────────────────────────────

def run_variant_a() -> None:
    """Вариант A: агент на smolagents с инструментом поиска по каталогу."""
    try:
        from smolagents import tool, CodeAgent, OpenAIServerModel
    except ImportError as e:
        raise ImportError(
            "Для Варианта A требуется пакет smolagents. "
            "Установите его: pip install smolagents"
        ) from e

    @tool
    def search_catalog(query: str) -> str:
        """Ищет товары в каталоге по запросу.

        Args:
            query: Поисковый запрос (название или часть названия товара).

        Returns:
            Строка с найденными товарами и их описанием,
            либо 'Товары не найдены', если ничего не найдено.
        """
        query_lower = query.lower()
        found = []
        for name, desc in CATALOG.items():
            if query_lower in name.lower():
                found.append(f"{name}: {desc}")
        if found:
            return "\n".join(found)
        return "Товары не найдены"

    load_env_file()
    api_key, base_url, model_name = get_api_config()

    model = OpenAIServerModel(
        model_id=model_name,
        api_base=base_url,
        api_key=api_key,
    )

    agent = CodeAgent(
        tools=[search_catalog],
        model=model,
    )

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Вариант A: smolagents CodeAgent готов")

    test_queries = [
        "Какие наушники есть в магазине?",
        "Есть ли у вас что-то от Apple дешевле 30 000?",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Запрос: {query}")
        print(f"{'='*60}")
        result = agent.run(query)
        print(f"\nОтвет:\n{result}")


# ── Вариант B: Structured Output ───────────────────────────────────────

class SearchPlan(BaseModel):
    """План поиска товаров в каталоге."""

    reasoning: str = Field(
        description="Объяснение, какие товары нужно найти и почему"
    )
    queries: list[str] = Field(
        description="Список поисковых запросов к каталогу"
    )


def search_catalog_simple(query: str) -> list[str]:
    """Простой поиск по каталогу (без декоратора @tool)."""
    query_lower = query.lower()
    results = []
    for name, desc in CATALOG.items():
        if query_lower in name.lower():
            results.append(f"{name}: {desc}")
    return results


def run_structured_agent(user_query: str, client: OpenAI, model_name: str) -> str:
    """Вариант B: Structured Output агент.

    Args:
        user_query: Запрос пользователя.
        client: Инициализированный клиент OpenAI.
        model_name: Название модели.

    Returns:
        Финальный текстовый ответ агента.
    """
    # Шаг 1: Получаем план поиска от модели
    system_prompt = (
        "Ты — консультант магазина электроники. "
        "Определи, какие товары нужно найти в каталоге, чтобы ответить на вопрос клиента. "
        "Сформируй поисковые запросы (ключевые слова или названия товаров)."
    )

    completion = client.beta.chat.completions.parse(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
        response_format=SearchPlan,
    )

    if not completion.choices or not completion.choices[0].message.parsed:
        raise ValueError("Модель вернула пустой ответ.")

    search_plan = completion.choices[0].message.parsed
    print(f"План поиска: {search_plan.reasoning}")
    print(f"Запросы: {search_plan.queries}")

    # Шаг 2: Ищем товары в каталоге по каждому запросу
    catalog_results = []
    for query in search_plan.queries:
        results = search_catalog_simple(query)
        if results:
            catalog_results.extend(results)

    if not catalog_results:
        catalog_results = ["Товары не найдены"]

    catalog_context = "\n".join(catalog_results)

    # Шаг 3: Отправляем результаты модели для финального ответа
    final_prompt = (
        f"Вопрос клиента: {user_query}\n\n"
        f"Найденные товары в каталоге:\n{catalog_context}\n\n"
        "Ответь на вопрос клиента, используя только информацию из каталога. "
        "Будь вежлив и лаконичен."
    )

    final_response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "Ты — консультант магазина электроники."},
            {"role": "user", "content": final_prompt},
        ],
    )

    return final_response.choices[0].message.content


def run_variant_b() -> None:
    """Вариант B: Structured Output агент."""
    load_env_file()
    api_key, base_url, model_name = get_api_config()
    client = OpenAI(api_key=api_key, base_url=base_url)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Вариант B: Structured Output агент готов")

    test_queries = [
        "Какие наушники есть в магазине?",
        "Есть ли у вас что-то от Apple дешевле 30 000?",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Запрос: {query}")
        print(f"{'='*60}")
        answer = run_structured_agent(query, client, model_name)
        print(f"\nОтвет:\n{answer}")


# ── Основная логика ────────────────────────────────────────────────────

def main() -> None:
    """Основная функция: запускает оба варианта агента."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    print("=" * 60)
    print("Function Calling: Агент-помощник магазина электроники")
    print("=" * 60)

    # Вариант A: smolagents
    print("\n--- Вариант A: smolagents CodeAgent ---")
    try:
        run_variant_a()
    except Exception as e:
        logger.error("Ошибка в Варианте A: %s", e)
        print(f"Вариант A не выполнен: {e}")

    # Вариант B: Structured Output
    print("\n--- Вариант B: Structured Output (Pydantic) ---")
    try:
        run_variant_b()
    except Exception as e:
        logger.error("Ошибка в Варианте B: %s", e)
        print(f"Вариант B не выполнен: {e}")


if __name__ == "__main__":
    main()
