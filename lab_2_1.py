"""Модуль для классификации новостей с использованием языковой модели.

Загружает переменные окружения, инициализирует клиент OpenAI и
оценивает точность классификации новостей по заданным категориям.
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


def check_classification(
    current_answer: str,
    target_answer: str,
    difficulty_level: int = 0,
) -> bool:
    """Проверяет, совпадает ли ответ модели с ожидаемым.

    При лёгком уровне сложности (0) проверяется вхождение целевой категории
    в ответ модели. При сложном уровне (1) требуется точное совпадение.

    Args:
        current_answer: Ответ, полученный от модели.
        target_answer: Ожидаемый правильный ответ.
        difficulty_level: Уровень сложности (0 — лёгкий, 1 — сложный).

    Returns:
        True, если ответ считается правильным, иначе False.
    """
    if difficulty_level == 0:
        return target_answer.lower() in current_answer.lower()
    return target_answer.lower() == current_answer.lower()


NEWS_TEXTS: list[str] = [
    "Премьер-министр Великобритании объявил о новых экологических "
    "нормах для борьбы с изменением климата.",
    "Команда Реал Мадрид выиграла Лигу чемпионов.",
    "Apple выпустила новый iPhone с улучшенной камерой.",
    "Цены на нефть упали на 5% из-за снижения мирового спроса.",
    "Европейский парламент принял новый климатический закон.",
    "Теннисист Новак Джокович выиграл Открытый чемпионат Австралии.",
    "Google представил ИИ для генерации изображений.",
]

TRUE_LABELS: list[str] = [
    "Политика",
    "Спорт",
    "Технологии",
    "Экономика",
    "Политика",
    "Спорт",
    "Технологии",
]

DIFFICULTY_LEVEL: int = 0

SYSTEM_PROMPT = (
    "Ты - классификатор новостей. Твоя задача - определить категорию "
    "новости из следующих: Политика, Спорт, Экономика, Технологии. "
    "Выведи только одно слово - название категории. Не добавляй никаких "
    "дополнительных слов, знаков препинания или пояснений. Например, "
    "если новость о политике, ответь: Политика."
)


def main() -> None:
    """Основная функция для оценки точности классификации новостей."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    load_env_file()
    api_key, base_url, model_name = get_api_config()
    client = OpenAI(api_key=api_key, base_url=base_url)

    correct_count = 0

    for index, news in enumerate(NEWS_TEXTS):
        try:
            answer = generate_answer(
                client=client,
                model_name=model_name,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=news,
            )
        except Exception as e:
            logger.error("Ошибка при обработке новости %d: %s", index, e)
            continue

        is_correct = check_classification(
            current_answer=answer,
            target_answer=TRUE_LABELS[index],
            difficulty_level=DIFFICULTY_LEVEL,
        )

        logger.info(
            "Новость %d: ответ модели='%s', ожидалось='%s', результат=%s",
            index,
            answer,
            TRUE_LABELS[index],
            is_correct,
        )

        print()
        print("=" * 20)
        print(f"Generate answer: {answer}")
        print(f"Real answer: {TRUE_LABELS[index]}")
        if is_correct:
            correct_count += 1
            print("CORRECTLY")
        else:
            print("WRONG")

    accuracy = round(correct_count / len(TRUE_LABELS), 2)
    print(f"\nAccuracy = {accuracy}")


if __name__ == "__main__":
    main()
