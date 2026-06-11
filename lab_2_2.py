"""Модуль для проверки работы модели с использованием Chain of Thought.

Загружает переменные окружения, инициализирует клиент OpenAI и
демонстрирует пошаговое рассуждение модели на логических задачах.
"""

import logging
import os
from pathlib import Path
from openai import OpenAI


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
        ValueError: Если не все обязательные переменные окружения заданы.
    """
    api_key = os.environ.get("API_KEY")
    base_url = os.environ.get("BASE_URL")
    model_name = os.environ.get("MODEL_NAME")

    if not all((api_key, base_url, model_name)):
        raise ValueError(
            "Не все обязательные переменные окружения заданы. "
            "Проверьте файл .env"
        )
    return api_key, base_url, model_name


def generate_answer(
    client: OpenAI,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """Генерирует ответ модели на основе системного и пользовательского промптов.

    Args:
        client: Инициализированный клиент OpenAI.
        model_name: Название модели для использования.
        system_prompt: Системный промпт, задающий поведение модели.
        user_prompt: Пользовательский запрос.

    Returns:
        Ответ модели, очищенный от пробелов по краям.

    Raises:
        ValueError: Если модель вернула пустой ответ.
        openai.OpenAIError: При ошибке обращения к API.
    """
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    if not response.choices or not response.choices[0].message.content:
        raise ValueError("Модель вернула пустой ответ.")
    return response.choices[0].message.content.strip()


SYSTEM_PROMPT = (
    "Ты — логический помощник. Перед тем как дать ответ, ты должен подробно "
    "рассуждать шаг за шагом. Строго придерживайся следующего формата вывода:\n"
    "Шаг 1: <первое логическое рассуждение>\n"
    "Шаг 2: <второе рассуждение>\n"
    "...\n"
    "Ответ: <финальный ответ кратко>\n\n"
    "Не добавляй никаких лишних слов, не отклоняйся от формата. Каждый шаг "
    "должен быть на новой строке, начинаться со слова 'Шаг N:' и содержать "
    "пояснение. В конце обязательно напиши 'Ответ:' и сам ответ."
)

TASKS: list[str] = [
    "У Маши было 5 яблок. Она отдала 2 яблока Пете, а потом нашла ещё 3. "
    "Сколько яблок у Маши теперь?",
    "Если все люди смертны, а Сократ — человек, то смертен ли Сократ?",
    "В магазине скидка 20% на товар стоимостью 1500 рублей. "
    "Сколько будет стоить товар со скидкой?",
    "У вас есть 3-литровое ведро и 5-литровое ведро. "
    "Как с их помощью отмерить ровно 4 литра воды?",
]


def main() -> None:
    """Основная функция для проверки работы модели на логических задачах."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    load_env_file()
    api_key, base_url, model_name = get_api_config()
    client = OpenAI(api_key=api_key, base_url=base_url)

    for task_number, task in enumerate(TASKS, 1):
        try:
            answer = generate_answer(
                client=client,
                model_name=model_name,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=task,
            )
        except Exception as e:
            logger.error("Ошибка при обработке задачи %d: %s", task_number, e)
            continue

        logger.info(
            "Задача %d: вопрос='%s', ответ модели:\n%s",
            task_number,
            task,
            answer,
        )

        print(f"\n--- Задача {task_number} ---")
        print(f"Вопрос: {task}")
        print(f"Ответ модели:\n{answer}")
        print("-" * 40)


if __name__ == "__main__":
    main()
