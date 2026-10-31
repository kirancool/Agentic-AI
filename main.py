import os
import requests
import streamlit as st

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.memory import ConversationBufferMemory
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
# =====================================================
# Load Environment Variables
# =====================================================

load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

WEATHER_BASE_URL = "http://api.weatherapi.com/v1"
TIMEOUT = 15

# =====================================================
# Weather API Helper
# =====================================================

def get_weather(endpoint: str, parameters: dict) -> dict:

    response = requests.get(
        f"{WEATHER_BASE_URL}/{endpoint}",
        params={
            "key": WEATHER_API_KEY,
            **parameters,
        },
        timeout=TIMEOUT,
    )

    response.raise_for_status()

    return response.json()

# =====================================================
# Weather Tools
# =====================================================

@tool
def get_current_weather(location: str) -> str:
    """
    Get current weather for a city.
    """

    try:
        data = get_weather(
            "current.json",
            {
                "q": location,
                "aqi": "no"
            }
        )

    except requests.HTTPError:
        return f"Weather data not found for '{location}'."

    except requests.RequestException as e:
        return f"Weather API Error: {str(e)}"

    loc = data["location"]
    cur = data["current"]

    return (
        f"Current Weather in {loc['name']}, "
        f"{loc['region']}, {loc['country']}\n\n"
        f"Condition: {cur['condition']['text']}\n"
        f"Temperature: {cur['temp_c']}°C ({cur['temp_f']}°F)\n"
        f"Humidity: {cur['humidity']}%\n"
        f"Wind: {cur['wind_kph']} kph\n"
        f"UV Index: {cur['uv']}\n"
        f"Local Time: {loc['localtime']}"
    )

@tool
def get_weather_forecast(location: str, days: int = 3) -> str:
    """
    Get weather forecast for a city.
    """

    days = max(1, min(days, 7))

    try:

        data = get_weather(
            "forecast.json",
            {
                "q": location,
                "days": days,
                "aqi": "no"
            }
        )

    except requests.HTTPError:
        return f"Forecast not found for '{location}'."

    except requests.RequestException as e:
        return f"Weather API Error: {str(e)}"

    loc = data["location"]

    result = [
        f"{days}-Day Forecast for "
        f"{loc['name']}, {loc['country']}"
    ]

    for day in data["forecast"]["forecastday"]:

        d = day["day"]

        result.append(
            f"""
Date: {day['date']}
Condition: {d['condition']['text']}
Max Temp: {d['maxtemp_c']}°C
Min Temp: {d['mintemp_c']}°C
Humidity: {d['avghumidity']}%
Chance of Rain: {d.get('daily_chance_of_rain', 'N/A')}%
UV: {d['uv']}
"""
        )

    return "\n".join(result)

# =====================================================
# Tavily Search Tool
# =====================================================

web_search_tool = TavilySearch(
    max_results=5,
    topic="general",
    search_depth="basic"
)

TOOLS = [
    get_current_weather,
    get_weather_forecast,
    web_search_tool
]

# =====================================================
# LLM
# =====================================================

llm = ChatNVIDIA(
    model="nvidia/nemotron-3-ultra-550b-a55b",
    api_key=NVIDIA_API_KEY,
    temperature=0.7,
    top_p=0.95,
    max_tokens=4096
)

# llm= ChatGoogleGenerativeAI(
#     model="gemma-4-31b-it",
#     api_key=GOOGLE_API_KEY,
#     temperature=0.7,
#     max_output_tokens=1024
# )

# =====================================================
# Prompt
# =====================================================

SYSTEM_PROMPT = """
You are a helpful AI assistant.

Rules:

1. Weather questions:
   - Use weather tools.
   - Give clear weather information.

2. General knowledge:
   - Use web search tool when needed.

3. Conversation:
   - Remember previous chat messages.

4. If no information is found:
   - Say you could not find the information.

5. Keep responses concise and helpful.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# =====================================================
# Memory
# =====================================================

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# =====================================================
# Agent
# =====================================================

agent = create_tool_calling_agent(
    llm=llm,
    tools=TOOLS,
    prompt=prompt,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=TOOLS,
    memory=memory,
    verbose=True,
)

# =====================================================
# Streamlit UI
# =====================================================

st.set_page_config(
    page_title="Weather & Search Assistant",
    page_icon="🌤️",
    layout="wide"
)

st.title("🌤️ Weather + Web Search Assistant")

st.markdown(
    """
Ask:
- Current weather
- Weather forecast
- General web search questions
"""
)

# Session History

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display History

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input

user_input = st.chat_input(
    "Ask me about weather or anything..."
)

if user_input:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                response = agent_executor.invoke(
                    {
                        "input": user_input
                    }
                )

                answer = response["output"]

            except Exception as e:
                answer = f"Error: {str(e)}"

            st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )