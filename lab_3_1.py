"""Модуль для извлечения структурированной информации из текста.

Использует языковую модель с Pydantic-схемой для извлечения
контактных данных из произвольного текста.
"""

import logging
import os
from pathlib import Path
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional


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


class ContactCard(BaseModel):
    """Карточка контакта с рассуждением и данными."""

    reasoning: str = Field(
        description="Пошаговое рассуждение: как были извлечены имя, "
        "email, телефон, компания и должность"
    )
    full_name: str = Field(description="Полное имя человека")
    email: str = Field(description="Электронная почта")
    phone: Optional[str] = Field(
        default=None, description="Номер телефона (если указан)"
    )
    company: str = Field(description="Название компании")
    position: str = Field(description="Должность")


def extract_contact_info(
    client: OpenAI,
    model_name: str,
    text: str,
) -> ContactCard:
    """Извлекает контактную информацию из текста.

    Args:
        client: Инициализированный клиент OpenAI.
        model_name: Название модели для использования.
        text: Текст, из которого нужно извлечь контактные данные.

    Returns:
        Объект ContactCard с извлечённой информацией.

    Raises:
        ValueError: Если модель вернула пустой ответ.
        openai.OpenAIError: При ошибке обращения к API.
    """
    system_prompt = (
        "Ты — извлекатель информации по заданной схеме. "
        "Сначала подумай шаг за шагом (поле reasoning), затем заполни "
        "остальные поля. Будь точен и используй только информацию из текста."
    )

    completion = client.beta.chat.completions.parse(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Текст: {text}"},
        ],
        response_format=ContactCard,
    )

    if not completion.choices or not completion.choices[0].message.parsed:
        raise ValueError("Модель вернула пустой ответ.")

    return completion.choices[0].message.parsed


def main() -> None:
    """Основная функция для извлечения контактных данных из текста."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    load_env_file()
    api_key, base_url, model_name = get_api_config()
    client = OpenAI(api_key=api_key, base_url=base_url)

    text_1 = (
        "Привет! Я Марина Соколова, работаю продакт-менеджером в Яндексе. "
        "Пишите мне на marina.sokolova@yandex.ru или звоните: "
        "+7 (916) 123-45-67."
    )

    try:
        result = extract_contact_info(
            client=client,
            model_name=model_name,
            text=text_1,
        )
    except Exception as e:
        logger.error("Ошибка при извлечении контактных данных: %s", e)
        raise

    logger.info(
        "Извлечены данные: имя='%s', email='%s', телефон='%s', "
        "компания='%s', должность='%s'",
        result.full_name,
        result.email,
        result.phone,
        result.company,
        result.position,
    )

    print("\n=== Результат извлечения ===\n")
    print(f"Рассуждение:\n{result.reasoning}\n")
    print(f"Имя: {result.full_name}")
    print(f"Email: {result.email}")
    print(f"Телефон: {result.phone}")
    print(f"Компания: {result.company}")
    print(f"Должность: {result.position}")


if __name__ == "__main__":
    main()
