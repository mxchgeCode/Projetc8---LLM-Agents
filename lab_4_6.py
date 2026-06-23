"""Модуль для демонстрации архитектур агентов на LangGraph.

Содержит 4 варианта реализации агента-помощника магазина электроники:
- Вариант A: Single Agent (LLM + Tools)
- Вариант B: Workflow Agent (interpreter → researcher → writer)
- Вариант C: Parallel Agent (dispatcher → 3 researchers → aggregator)
- Вариант D: ReAct Agent (create_agent)
"""

import logging
import os
from pathlib import Path
from typing import TypedDict

from openai import OpenAI
from pydantic import BaseModel, Field

# LangGraph / LangChain imports
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool


# ── Данные (каталог товаров) ───────────────────────────────────────────

PRODUCTS = {
    "iPhone 16 Pro": "Флагман Apple с чипом A18 Pro, 8 ГБ RAM, 256 ГБ, камера 48 МП, титановый корпус.",
    "Samsung Galaxy S25 Ultra": "Топовый Android с S Pen, 12 ГБ RAM, 200 МП камера, 5000 мА·ч, Snapdragon 8 Elite.",
    "Xiaomi 15": "Флагман с камерой Leica, 12 ГБ RAM, 5400 мА·ч, Snapdragon 8 Elite, привлекательная цена.",
    "Google Pixel 9": "Лучшая камера на Android, чистый софт, 7 лет обновлений. (нет в наличии)",
    "MacBook Air M3": "Тонкий лёгкий ноутбук с Apple M3, 16 ГБ RAM, автономность до 18 ч, без вентилятора.",
    "ASUS ROG Strix G16": "Игровой ноутбук: Intel i7, RTX 4060, 16 165 Гц, 1 ТБ SSD.",
    "Lenovo ThinkPad X1 Carbon Gen 12": "Бизнес-ультрабук с OLED 2.8K, 32 ГБ RAM, Intel Ultra 7.",
    "Sony WH-1000XM5": "Лучшее шумоподавление, премиальный звук, 30 ч работы, LDAC.",
    "AirPods Pro 2": "TWS с адаптивным шумоподавлением, 6 ч работы, интеграция с Apple.",
    "JBL Tune 520BT": "Бюджетные on-ear наушники, 57 ч работы, фирменный звук JBL.",
    "iPad Air M2": "Планшет с чипом M2, 11 Liquid Retina, поддержка Apple Pencil Pro.",
    "Samsung Galaxy Tab S9 FE": "Доступный планшет с S Pen в комплекте, защита IP68.",
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


# ── Инструмент поиска по каталогу ──────────────────────────────────────

@tool
def search_products(query: str) -> str:
    """Ищет товары в каталоге по запросу.

    Args:
        query: Поисковый запрос (название или часть названия товара).

    Returns:
        Строка с найденными товарами и их описанием,
        либо 'Товары не найдены', если ничего не найдено.
    """
    query_lower = query.lower()
    found = []
    for name, desc in PRODUCTS.items():
        if query_lower in name.lower():
            found.append(f"{name}: {desc}")
    if found:
        return "\n".join(found)
    return "Товары не найдены"


def search_products_simple(query: str) -> list[str]:
    """Простой поиск по каталогу (без декоратора @tool)."""
    query_lower = query.lower()
    results = []
    for name, desc in PRODUCTS.items():
        if query_lower in name.lower():
            results.append(f"{name}: {desc}")
    return results


def tool_call(tool_answer: dict, tools_map: dict) -> str:
    """Вызывает инструмент по результату tool_call."""
    return tools_map[tool_answer["name"]].invoke(tool_answer["args"])


# ── Вариант A: Single Agent ─────────────────────────────────────────────

def run_single_agent(query: str, llm_with_tools, tools_map: dict) -> str:
    """Single Agent: LLM + Tools."""
    response = llm_with_tools.invoke(query)

    if response.tool_calls:
        tools_result = ""
        for tool_response in response.tool_calls:
            tools_result += f"{tool_response['name']}: {tool_call(tool_response, tools_map)}\n"

        answer_response = llm_with_tools.invoke(
            f"Запрос: {query}. Результат вызова инструментов: {tools_result}. Дай ответ пользователю"
        )
        return answer_response.content

    return response.content


def run_variant_a() -> None:
    """Вариант A: Single Agent."""
    load_env_file()
    api_key, base_url, model_name = get_api_config()

    llm = ChatOpenAI(api_key=api_key, base_url=base_url, model=model_name)
    llm_with_tools = llm.bind_tools([search_products])

    tools_map = {"search_products": search_products}

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Вариант A: Single Agent готов")

    test_queries = [
        "Какие наушники есть в магазине?",
        "Есть ли у вас что-то от Apple дешевле 30 000?",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Запрос: {query}")
        print(f"{'='*60}")
        result = run_single_agent(query, llm_with_tools, tools_map)
        print(f"\nОтвет:\n{result}")


# ── Вариант B: Workflow Agent ───────────────────────────────────────────

class WorkflowState(TypedDict):
    user_query: str
    interpreted_query: str
    search_results: str
    final_answer: str


def interpreter(state: WorkflowState, llm: ChatOpenAI) -> dict:
    """Шаг 1: LLM упрощает и интерпретирует запрос."""
    print("Called interpreter")
    response = llm.invoke(
        f"Упрости и интерпретируй следующий запрос пользователя. "
        f"Запрос: {state['user_query']}"
    )
    return {"interpreted_query": response.content}


def researcher(state: WorkflowState, llm_with_tools, tools_map: dict) -> dict:
    """Шаг 2: Агент с search_tool — сам ищет информацию."""
    print("Called researcher")
    result = run_single_agent(
        f"Найди информацию по теме: {state['interpreted_query']}",
        llm_with_tools,
        tools_map,
    )
    return {"search_results": result}


def writer(state: WorkflowState, llm: ChatOpenAI) -> dict:
    """Шаг 3: LLM — генерирует финальный ответ."""
    print("Called writer")
    response = llm.invoke(
        f"На основе результатов поиска ответь на вопрос пользователя.\n\n"
        f"Вопрос: {state['user_query']}\n\n"
        f"Результаты поиска:\n{state['search_results']}"
    )
    return {"final_answer": response.content}


def run_variant_b() -> None:
    """Вариант B: Workflow Agent."""
    load_env_file()
    api_key, base_url, model_name = get_api_config()

    llm = ChatOpenAI(api_key=api_key, base_url=base_url, model=model_name)
    llm_with_tools = llm.bind_tools([search_products])
    tools_map = {"search_products": search_products}

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Вариант B: Workflow Agent готов")

    # Создаем граф
    workflow = StateGraph(WorkflowState)
    workflow.add_node("interpreter", lambda s: interpreter(s, llm))
    workflow.add_node("researcher", lambda s: researcher(s, llm_with_tools, tools_map))
    workflow.add_node("writer", lambda s: writer(s, llm))

    workflow.set_entry_point("interpreter")
    workflow.add_edge("interpreter", "researcher")
    workflow.add_edge("researcher", "writer")
    workflow.add_edge("writer", END)

    workflow_agent = workflow.compile()

    test_queries = [
        "Какие наушники есть в магазине?",
        "Есть ли у вас что-то от Apple дешевле 30 000?",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Запрос: {query}")
        print(f"{'='*60}")
        response = workflow_agent.invoke({"user_query": query})
        print(f"\nОтвет:\n{response['final_answer']}")


# ── Вариант C: Parallel Agent ──────────────────────────────────────────

class DispatcherTasks(BaseModel):
    """План поисковых запросов для параллельных агентов."""

    task_1: str = Field(description="Первый уникальный поисковый запрос")
    task_2: str = Field(description="Второй уникальный поисковый запрос")
    task_3: str = Field(description="Третий уникальный поисковый запрос")


class ParallelState(TypedDict):
    user_query: str
    task_1: str
    task_2: str
    task_3: str
    result_1: str
    result_2: str
    result_3: str
    final_answer: str


def dispatcher(state: ParallelState, dispatcher_llm) -> dict:
    """Шаг 1: LLM-диспетчер сама решает, какие запросы отправить агентам."""
    print("Called dispatcher")

    prompt = (
        f"Тебе нужно исследовать вопрос пользователя с помощью 3 параллельных агентов. "
        f"Придумай для каждого свой уникальный поисковый запрос, чтобы максимально полно раскрыть тему.\n\n"
        f"Вопрос пользователя: {state['user_query']}"
    )

    tasks = dispatcher_llm.invoke(prompt)

    return {
        "task_1": tasks.task_1,
        "task_2": tasks.task_2,
        "task_3": tasks.task_3,
    }


def researcher_1(state: ParallelState, llm_with_tools, tools_map: dict) -> dict:
    """Первый параллельный исследователь."""
    print("Called researcher_1")
    summary = run_single_agent(
        f"Произведи поиск по запросу: '{state['task_1']}'. После кратко обобщи результаты поиска",
        llm_with_tools,
        tools_map,
    )
    return {"result_1": summary}


def researcher_2(state: ParallelState, llm_with_tools, tools_map: dict) -> dict:
    """Второй параллельный исследователь."""
    print("Called researcher_2")
    summary = run_single_agent(
        f"Произведи поиск по запросу: '{state['task_2']}'. После кратко обобщи результаты поиска",
        llm_with_tools,
        tools_map,
    )
    return {"result_2": summary}


def researcher_3(state: ParallelState, llm_with_tools, tools_map: dict) -> dict:
    """Третий параллельный исследователь."""
    print("Called researcher_3")
    summary = run_single_agent(
        f"Произведи поиск по запросу: '{state['task_3']}'. После кратко обобщи результаты поиска",
        llm_with_tools,
        tools_map,
    )
    return {"result_3": summary}


def aggregator(state: ParallelState, llm: ChatOpenAI) -> dict:
    """Агрегатор: объединяет результаты трёх исследований."""
    print("Called aggregator")
    response = llm.invoke(
        f"Объедини результаты трёх исследований в один связный ответ на русском языке.\n\n"
        f"Вопрос пользователя: {state['user_query']}\n\n"
        f"Исследование 1:\n{state['result_1']}\n\n"
        f"Исследование 2:\n{state['result_2']}\n\n"
        f"Исследование 3:\n{state['result_3']}\n\n"
        f"Дай полный структурированный ответ."
    )
    return {"final_answer": response.content}


def run_variant_c() -> None:
    """Вариант C: Parallel Agent."""
    load_env_file()
    api_key, base_url, model_name = get_api_config()

    llm = ChatOpenAI(api_key=api_key, base_url=base_url, model=model_name)
    llm_with_tools = llm.bind_tools([search_products])
    tools_map = {"search_products": search_products}

    # Создаем structured output LLM для диспетчера
    dispatcher_llm = llm.with_structured_output(DispatcherTasks)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Вариант C: Parallel Agent готов")

    # Создаем граф
    parallel = StateGraph(ParallelState)
    parallel.add_node("dispatcher", lambda s: dispatcher(s, dispatcher_llm))
    parallel.add_node("researcher_1", lambda s: researcher_1(s, llm_with_tools, tools_map))
    parallel.add_node("researcher_2", lambda s: researcher_2(s, llm_with_tools, tools_map))
    parallel.add_node("researcher_3", lambda s: researcher_3(s, llm_with_tools, tools_map))
    parallel.add_node("aggregator", lambda s: aggregator(s, llm))

    parallel.set_entry_point("dispatcher")
    parallel.add_edge("dispatcher", "researcher_1")
    parallel.add_edge("dispatcher", "researcher_2")
    parallel.add_edge("dispatcher", "researcher_3")
    parallel.add_edge("researcher_1", "aggregator")
    parallel.add_edge("researcher_2", "aggregator")
    parallel.add_edge("researcher_3", "aggregator")
    parallel.add_edge("aggregator", END)

    parallel_agent = parallel.compile()

    test_queries = [
        "Сравни iPhone 16 Pro, Samsung Galaxy S25 Ultra и Xiaomi 15",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Запрос: {query}")
        print(f"{'='*60}")
        result = parallel_agent.invoke({"user_query": query})
        print(f"\nОтвет:\n{result['final_answer']}")


# ── Вариант D: ReAct Agent ─────────────────────────────────────────────

def run_variant_d() -> None:
    """Вариант D: ReAct Agent."""
    load_env_file()
    api_key, base_url, model_name = get_api_config()

    llm = ChatOpenAI(api_key=api_key, base_url=base_url, model=model_name)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Вариант D: ReAct Agent готов")

    system_prompt = (
        "Ты — полезный ассистент магазина электроники. "
        "Используй поиск товаров, чтобы находить информацию о наличии и описании товаров. "
        "Рассуждай пошагово. Отвечай на русском языке."
    )

    react_agent = create_react_agent(
        llm,
        [search_products],
        prompt=system_prompt,
    )

    test_queries = [
        "Какие наушники есть в магазине?",
        "Есть ли у вас что-то от Apple дешевле 30 000?",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Запрос: {query}")
        print(f"{'='*60}")
        result = react_agent.invoke({"messages": [("user", query)]})
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


# ── Основная логика ────────────────────────────────────────────────────

def main() -> None:
    """Основная функция: запускает все 4 варианта агента."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    print("=" * 60)
    print("Архитектуры агентов на LangGraph: Агент-помощник магазина электроники")
    print("=" * 60)

    # Вариант A: Single Agent
    print("\n--- Вариант A: Single Agent ---")
    try:
        run_variant_a()
    except Exception as e:
        logger.error("Ошибка в Варианте A: %s", e)
        print(f"Вариант A не выполнен: {e}")

    # Вариант B: Workflow Agent
    print("\n--- Вариант B: Workflow Agent ---")
    try:
        run_variant_b()
    except Exception as e:
        logger.error("Ошибка в Варианте B: %s", e)
        print(f"Вариант B не выполнен: {e}")

    # Вариант C: Parallel Agent
    print("\n--- Вариант C: Parallel Agent ---")
    try:
        run_variant_c()
    except Exception as e:
        logger.error("Ошибка в Варианте C: %s", e)
        print(f"Вариант C не выполнен: {e}")

    # Вариант D: ReAct Agent
    print("\n--- Вариант D: ReAct Agent ---")
    try:
        run_variant_d()
    except Exception as e:
        logger.error("Ошибка в Варианте D: %s", e)
        print(f"Вариант D не выполнен: {e}")


if __name__ == "__main__":
    main()
