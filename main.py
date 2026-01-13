import os
import requests

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain_ollama.chat_models import ChatOllama
from langfuse.client import Langfuse

load_dotenv()

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)

llm = ChatOllama(model="llama3.2")


@tool
def calculate(expression: str) -> str:
    """
    Use this tool to perform any mathematical operation.
    You MUST use this tool whenever you need to add, subtract, multiply or divide numbers.
    Do NOT calculate in your head - always use this tool.

    Input: A math expression with numbers and operators only.
    Examples: '5.36 * 2', '100 + 50', '10 / 3'

    Returns: The numeric result as a string.
    """
    try:
        return str(eval(expression))
    except (SyntaxError, NameError, TypeError, ZeroDivisionError, ValueError) as e:
        return f"Error: {e}"


@tool
def get_dollar_rate() -> str:
    """
    Use this tool to get the current USD to BRL exchange rate.
    Call this when the user asks about dollar price or exchange rate.

    Input: None required.

    Returns: The current rate as a decimal number (e.g., '5.36874').
    """
    response = requests.get("https://economia.awesomeapi.com.br/last/USD-BRL", timeout=20)
    return response.json()["USDBRL"]["bid"]


tools = [calculate, get_dollar_rate]
tools_by_name = {t.name: t for t in tools}

llm_with_tools = llm.bind_tools(tools)

prompt = SystemMessage(
    content=(
        "In the response assign the response with : Powered By Ollama and Jefferson."
        "You are a helpful assistant that can use tools to answer user questions. "
        "Use the provided tools whenever necessary to get accurate information."
        "You have access to the following tools:\n"
        + "\n".join(f"- {tool.name}: {tool.description}" for tool in tools)
    )
)

question = HumanMessage(
    content=(
        "Calculates 357 * 3? " #user input message
    )
)

trace = langfuse.trace(name="tool-calling", input=question.content)

generation = trace.generation(
    name="llm-first-call",
    input=[
        {"role": "system", "content": prompt.content},
        {"role": "user", "content": question.content}
    ],
)

response_with_tool = llm_with_tools.invoke([prompt, question])

generation.end(
    output=str(response_with_tool.tool_calls)
    if response_with_tool.tool_calls
    else response_with_tool.content
)

if response_with_tool.tool_calls:
    messages = [question, response_with_tool]

    for tool_call in response_with_tool.tool_calls:
        print(f"Tool: {tool_call['name']}, Args: {tool_call['args']}")

        span = trace.span(name=f"tool:{tool_call['name']}", input=tool_call["args"])
        tool_fn = tools_by_name[tool_call["name"]]
        result = tool_fn.invoke(tool_call["args"])
        span.end(output=result)

        messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))

    generation2 = trace.generation(name="llm-final-call", input=str(messages))
    final_response = llm_with_tools.invoke(messages)
    generation2.end(output=final_response.content)

    print("Response:", final_response.content)
    print("Tools used:", [tc["name"] for tc in response_with_tool.tool_calls])
    trace.update(output=final_response.content)
else:
    print("Response:", response_with_tool.content)
    trace.update(output=response_with_tool.content)

langfuse.flush()
print("Trace URL:", trace.get_trace_url())
