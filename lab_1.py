import os
from pathlib import Path
from openai import OpenAI


def load_env_file(file_path: str = ".env") -> None:
    """Загружает переменные окружения из файла .env.

    Args:
        file_path: Путь к файлу с переменными окружения.
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


# Загружаем переменные окружения
load_env_file()

API_KEY = os.environ.get("API_KEY")
BASE_URL = os.environ.get("BASE_URL")
MODEL_NAME = os.environ.get("MODEL_NAME")

if not all([API_KEY, BASE_URL, MODEL_NAME]):
    raise ValueError(
        "Не все обязательные переменные окружения заданы. "
        "Проверьте файл .env"
    )

system_prompt = (
    "Ты — полезный, дружелюбный и честный чат-бот. Отвечай кратко и по делу. "
    "Если не знаешь ответа — так и скажи. Не пиши ничего опасного, незаконного "
    "или оскорбительного. Не выдавай себя за человека. Всегда общайся на русском языке."
)

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# Список для хранения всей истории сообщений
messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

cnt = 5
while cnt > 0:
    user_input = input("> ")

    # Добавляем сообщение пользователя в историю
    messages.append({"role": "user", "content": user_input})

    # Отправляем ВСЮ историю в модель
    response = client.chat.completions.create(
        model=MODEL_NAME, messages=messages
    )

    # Получаем ответ ассистента
    assistant_answer = response.choices[0].message.content
    print(f"\n{assistant_answer}\n")

    # Добавляем ответ ассистента в историю
    messages.append(dict(role="assistant", content=assistant_answer))

    cnt -= 1
