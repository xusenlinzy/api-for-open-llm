import json
import math


def get_current_weather(location):
    key_selection = {
        "current_condition": ["temp_C", "FeelsLikeC", "humidity", "weatherDesc", "observation_time"],
    }
    import requests
    try:
        resp = requests.get(f"https://wttr.in/{location}?format=j1")
        resp.raise_for_status()
        resp = resp.json()
        ret = {k: {_v: resp[k][0][_v] for _v in v} for k, v in key_selection.items()}
    except:
        import traceback
        ret = "Error encountered while fetching weather data!\n" + traceback.format_exc()

    return json.dumps(ret)


def calculator(formula):
    formula = formula.replace("^", "**")
    if "sqrt" in formula:
        formula = formula.replace("sqrt", "math.sqrt")
    elif "log" in formula:
        formula = formula.replace("log", "math.log")
    return str(eval(formula))


available_functions = {
    "get_current_weather": get_current_weather,
    "calculator": calculator,
}

functions = [
    {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
            },
            "required": ["location"],
        },
    },
    {
        "name": "calculator",
        "description": "Useful for when you need to answer questions about math.",
        "parameters": {
            "type": "object",
            "properties": {
                "formula": {
                    "type": "string",
                    "description": "The formula to be calculated.",
                },
            },
            "required": ["formula"],
        },
    }
]


def postprocess_text(text: str) -> str:
    text = text.replace("\(", "$")
    text = text.replace("\)", "$")
    text = text.replace("\[", "$$")
    text = text.replace("\]", "$$")
    text = text.replace("<|assistant|>", "")
    text = text.replace("<|observation|>", "")
    text = text.replace("<|system|>", "")
    text = text.replace("<|user|>", "")
    return text.strip()
