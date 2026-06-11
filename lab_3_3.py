"""Модуль для анализа вакансий и матчинга с резюме.

Выполняет двухэтапный анализ:
Шаг A — извлечение структурированных данных из текста вакансии.
Шаг B — оценка соответствия вакансии и резюме кандидата.
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


class JobPosting(BaseModel):
    """Структурированное представление вакансии."""

    title: str = Field(description="Название позиции")
    company: str = Field(description="Название компании")
    required_skills: list[str] = Field(
        description="Список обязательных навыков"
    )
    nice_to_have_skills: list[str] = Field(
        description="Список желательных навыков"
    )
    experience_years_min: int = Field(
        description="Минимальный требуемый опыт в годах"
    )
    remote: Literal["yes", "no", "hybrid"] = Field(
        description="Формат работы: удалённо, офис или гибрид"
    )


class MatchResult(BaseModel):
    """Результат оценки соответствия вакансии и резюме."""

    reasoning: str = Field(
        description="Обоснование оценки соответствия"
    )
    match_score: int = Field(
        description="Оценка соответствия от 0 до 100",
        ge=0,
        le=100,
    )
    matched_skills: list[str] = Field(
        description="Навыки, которые совпали у кандидата и в вакансии"
    )
    missing_skills: list[str] = Field(
        description="Навыки, которых не хватает кандидату"
    )
    verdict: Literal[
        "strong_match", "partial_match", "weak_match"
    ] = Field(description="Итоговая оценка соответствия")


def extract_job_posting(
    client: OpenAI,
    model_name: str,
    text: str,
) -> JobPosting:
    """Извлекает структурированные данные из текста вакансии.

    Args:
        client: Инициализированный клиент OpenAI.
        model_name: Название модели для использования.
        text: Текст вакансии.

    Returns:
        Объект JobPosting с извлечённой информацией.

    Raises:
        ValueError: Если модель вернула пустой ответ.
        openai.OpenAIError: При ошибке обращения к API.
    """
    system_prompt = (
        "Ты — извлекатель информации о вакансиях. "
        "Сначала подумай шаг за шагом (поле reasoning), затем заполни "
        "остальные поля. Будь точен и используй только информацию из текста."
    )

    completion = client.beta.chat.completions.parse(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Вакансия: {text}"},
        ],
        response_format=JobPosting,
    )

    if not completion.choices or not completion.choices[0].message.parsed:
        raise ValueError("Модель вернула пустой ответ.")

    return completion.choices[0].message.parsed


def match_resume_to_job(
    client: OpenAI,
    model_name: str,
    job_posting: JobPosting,
    resume_text: str,
) -> MatchResult:
    """Оценивает соответствие резюме вакансии.

    Args:
        client: Инициализированный клиент OpenAI.
        model_name: Название модели для использования.
        job_posting: Структурированные данные вакансии.
        resume_text: Текст резюме кандидата.

    Returns:
        Объект MatchResult с оценкой соответствия.

    Raises:
        ValueError: Если модель вернула пустой ответ.
        openai.OpenAIError: При ошибке обращения к API.
    """
    system_prompt = (
        "Ты — HR-аналитик. Оцени соответствие резюме кандидата "
        "требованиям вакансии. Сначала подумай шаг за шагом "
        "(поле reasoning), затем заполни остальные поля."
    )

    prompt = (
        f"Вакансия:\n{job_posting.model_dump_json(indent=2)}\n\n"
        f"Резюме:\n{resume_text}"
    )

    completion = client.beta.chat.completions.parse(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        response_format=MatchResult,
    )

    if not completion.choices or not completion.choices[0].message.parsed:
        raise ValueError("Модель вернула пустой ответ.")

    return completion.choices[0].message.parsed


def main() -> None:
    """Основная функция для анализа вакансии и матчинга с резюме."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    load_env_file()
    api_key, base_url, model_name = get_api_config()
    client = OpenAI(api_key=api_key, base_url=base_url)

    job_posting_text = """
    Компания "DataFlow" ищет Middle Python-разработчика.
    Обязательно: Python 3+, FastAPI, PostgreSQL, Docker, Git.
    Будет плюсом: Kubernetes, CI/CD, Redis, опыт с ML-пайплайнами.
    Опыт от 3 лет. Гибридный формат (2 дня офис, 3 дня удалённо).
    """

    candidate_cv = """
    Иван Кузнецов, Python-разработчик, 4 года опыта.
    Стек: Python, Django, FastAPI, PostgreSQL, Docker, Git, Linux.
    Работал с Redis и Celery. Базово знаком с Kubernetes.
    Прошёл курс по ML, но коммерческого опыта с ML-пайплайнами нет.
    """

    try:
        job = extract_job_posting(
            client=client,
            model_name=model_name,
            text=job_posting_text,
        )
    except Exception as e:
        logger.error("Ошибка при извлечении данных вакансии: %s", e)
        raise

    logger.info(
        "Вакансия: %s в %s, удалённо: %s",
        job.title,
        job.company,
        job.remote,
    )

    print("\n=== Шаг A: Данные вакансии ===")
    print(f"Название: {job.title}")
    print(f"Компания: {job.company}")
    print(f"Опыт от: {job.experience_years_min} лет")
    print(f"Удалённо: {job.remote}")
    print(f"Обязательные навыки: {', '.join(job.required_skills)}")
    print(f"Желательные навыки: {', '.join(job.nice_to_have_skills)}")

    try:
        match = match_resume_to_job(
            client=client,
            model_name=model_name,
            job_posting=job,
            resume_text=candidate_cv,
        )
    except Exception as e:
        logger.error("Ошибка при матчинге резюме: %s", e)
        raise

    logger.info(
        "Матчинг: %s, оценка: %d/100",
        match.verdict,
        match.match_score,
    )

    print("\n=== Шаг B: Результат матчинга ===")
    print(f"Оценка: {match.match_score}/100")
    print(f"Вердикт: {match.verdict}")
    print(f"Совпавшие навыки: {', '.join(match.matched_skills)}")
    print(f"Не хватает: {', '.join(match.missing_skills)}")
    print(f"Обоснование: {match.reasoning}")


if __name__ == "__main__":
    main()
