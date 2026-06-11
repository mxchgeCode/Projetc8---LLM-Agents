"""Модуль для классификации обращений в поддержку.

Использует языковую модель с Pydantic-схемой для маршрутизации
тикетов по категориям и приоритетам.
"""

import logging
import os
from pathlib import Path
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Literal


def load_env_file(file_path: str = ".env") -> None:
    """Загружает переменные окружения из файла .env.

    Args:
        file_path: Путь к файлу с переменными окружения.

    Raises:
        FileNotFoundError: Если файл с переменными окружения не найден.
    """
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
    """Получает конфигурацию API из переменных окружения.

    Returns:
        Кортеж (API_KEY, BASE_URL, MODEL_NAME).

    Raises:
        ValueError: Если не все обязательные переменные окружения заданы
            или BASE_URL не содержит протокол.
    """
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


class TicketClassification(BaseModel):
    """Результат классификации обращения в поддержку."""

    reasoning: str = Field(
        description="Краткое объяснение, почему выбрана именно такая "
        "категория и приоритет"
    )
    category: Literal[
        "billing", "technical", "account", "feature_request", "other"
    ] = Field(description="Категория обращения")
    priority: Literal["low", "medium", "high", "critical"] = Field(
        description="Приоритет обращения"
    )
    suggested_reply: str = Field(
        description="Краткий ответ клиенту на основе обращения"
    )


def extract_ticket_classification(
    client: OpenAI,
    model_name: str,
    text: str,
) -> TicketClassification:
    """Классифицирует обращение в поддержку.

    Args:
        client: Инициализированный клиент OpenAI.
        model_name: Название модели для использования.
        text: Текст обращения для классификации.

    Returns:
        Объект TicketClassification с результатами классификации.

    Raises:
        ValueError: Если модель вернула пустой ответ.
        openai.OpenAIError: При ошибке обращения к API.
    """
    system_prompt = (
        "Ты — классификатор обращений в поддержку. "
        "Определи категорию, приоритет и предложи ответ клиенту. "
        "Сначала подумай шаг за шагом (поле reasoning), затем заполни "
        "остальные поля."
    )

    completion = client.beta.chat.completions.parse(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Обращение: {text}"},
        ],
        response_format=TicketClassification,
    )

    if not completion.choices or not completion.choices[0].message.parsed:
        raise ValueError("Модель вернула пустой ответ.")

    return completion.choices[0].message.parsed


def main() -> None:
    """Основная функция для классификации обращений в поддержку."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    load_env_file()
    api_key, base_url, model_name = get_api_config()
    client = OpenAI(api_key=api_key, base_url=base_url)

    tickets = [
        "У меня списали деньги дважды за подписку в этом месяце. "
        "Верните деньги!",
        "Приложение вылетает при загрузке фото на Android 14.",
        "Было бы здорово добавить тёмную тему.",
        "Не могу войти в аккаунт — пишет 'неверный пароль', "
        "хотя я его точно помню.",
        "Сайт полностью не работает, у нас из-за этого стоит продакшен!!!",
    ]

    for ticket in tickets:
        try:
            result = extract_ticket_classification(
                client=client,
                model_name=model_name,
                text=ticket,
            )
        except Exception as e:
            logger.error("Ошибка при классификации тикета: %s", e)
            raise

        logger.info(
            "Тикет: [%s | %s] %s",
            result.category,
            result.priority,
            ticket,
        )

        print(f"\n=== Тикет ===")
        print(f"Текст: {ticket}")
        print(f"Категория: {result.category}")
        print(f"Приоритет: {result.priority}")
        print(f"Рассуждение: {result.reasoning}")
        print(f"Предлагаемый ответ: {result.suggested_reply}")


if __name__ == "__main__":
    main()
