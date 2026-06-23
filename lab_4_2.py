"""Модуль для создания ReAct-агента консультанта магазина электроники.

Агент использует LangGraph и умеет:
- Искать товары в каталоге
- Проверять наличие товаров в городах
"""

import logging
import os
from pathlib import Path

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool


# ── Данные (каталог и остатки) ────────────────────────────────────────

CATALOG = {
    "iphone 15": {"price": 999, "category": "смартфон", "specs": "6.1 дюйм, A16 Bionic, 128GB"},
    "samsung s24": {"price": 899, "category": "смартфон", "specs": "6.2 дюйм, Snapdragon 8 Gen 3, 128GB"},
    "macbook air": {"price": 1299, "category": "ноутбук", "specs": "13.6 дюйм, Apple M3, 8GB RAM"},
    "dell xps 15": {"price": 1199, "category": "ноутбук", "specs": "15.6 дюйм, Intel i7, 16GB RAM"},
}

STOCK = {
    "iphone 15": {"москва": 5, "санкт-петербург": 2},
    "samsung s24": {"москва": 0, "санкт-петербург": 8},
    "macbook air": {"москва": 3, "санкт-петербург": 0},
    "dell xps 15": {"москва": 1, "санкт-петербург": 4},
}


# ── Инструменты ───────────────────────────────────────────────────────

@tool
def search_product(query: str) -> str:
    """Ищет товар в каталоге по запросу.

    Args:
        query: Поисковый запрос (название или часть названия товара).

    Returns:
        Строка с названием, ценой и характеристиками товара,
        либо 'Товар не найден', если товар отсутствует в каталоге.
    """
    query_lower = query.lower()
    for name, info in CATALOG.items():
        if query_lower in name:
            return (
                f"{name.title()} — {info['category']}, "
                f"цена: ${info['price']}, характеристики: {info['specs']}"
            )
    return "Товар не найден"


@tool
def check_stock(product: str, city: str) -> str:
    """Проверяет наличие товара в указанном городе.

    Args:
        product: Название товара (как в каталоге, например 'iphone 15').
        city: Название города (например, 'москва' или 'санкт-петербург').

    Returns:
        Количество товара в наличии в городе,
        либо 'нет в наличии', если товар отсутствует.
    """
    product_lower = product.lower()
    city_lower = city.lower()

    if product_lower not in STOCK:
        return "Товар не найден в каталоге"

    city_stock = STOCK[product_lower]
    if city_lower not in city_stock:
        return f"Нет данных по городу '{city}'"

    quantity = city_stock[city_lower]
    if quantity > 0:
        return f"В наличии: {quantity} шт. в городе {city.title()}"
    return "нет в наличии"


# ── Загрузка конфигурации ─────────────────────────────────────────────

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


# ── Основная логика ───────────────────────────────────────────────────

def main() -> None:
    """Создаёт и запускает ReAct-агента консультанта магазина электроники."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    load_env_file()
    api_key, base_url, model_name = get_api_config()

    model = ChatOpenAI(
        base_url=base_url,
        api_key=api_key,
        model=model_name,
    )

    system_prompt = (
        "Ты — консультант магазина электроники. "
        "Помогай клиентам с выбором товаров: ищи товары в каталоге, "
        "сообщай цены и характеристики, проверяй наличие в городах. "
        "Будь вежлив и лаконичен. "
        "Если товар не найден — честно сообщи об этом."
    )

    agent = create_react_agent(
        model,
        [search_product, check_stock],
        prompt=system_prompt,
    )

    logger.info("Агент создан. Запускаю тестовый запрос...")

    test_query = "Сколько стоит samsung и есть ли он в наличии в Москве?"
    logger.info("Тестовый запрос: %s", test_query)

    result = agent.invoke({"messages": [("user", test_query)]})

    print("\n=== Цепочка сообщений ===")
    for msg in result["messages"]:
        if msg.type == "human":
            print(f"User: {msg.content}")
        elif msg.type == "ai" and msg.tool_calls:
            calls = [f"{tc['name']}({tc['args']})" for tc in msg.tool_calls]
            print(f"AI вызывает: {', '.join(calls)}")
        elif msg.type == "tool":
            print(f"Tool [{msg.name}]: {msg.content}")
        elif msg.type == "ai":
            print(f"AI: {msg.content}")

    print("\n=== Финальный ответ агента ===")
    final_msg = result["messages"][-1]
    if hasattr(final_msg, "content") and final_msg.content:
        print(final_msg.content)
    else:
        print("(пустой ответ)")


if __name__ == "__main__":
    main()
