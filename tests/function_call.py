import math

import openai
from scipy import integrate

# Modify OpenAI's API key and API base to use vLLM's API server.
openai.api_key = "EMPTY"
openai.api_base = "http://192.168.0.53:7891/v1"


def calculate_quad(formula_str: str, a: float, b: float) -> float:
    """ 计算数值积分 """
    return integrate.quad(eval('lambda x: ' + formula_str), a, b)[0]


def calculate_sqrt(y: float) -> float:
    """ 计算平方根 """
    return math.sqrt(y)


functions = [
    {
        "name": "calculate_quad",
        "description": "calculate_quad是一个可以计算给定区间内函数定积分数值的工具。",
        "parameters": [
            {
                'name': 'formula_str',
                'description': '一个数学函数的表达式，例如x**2',
                'required': True,
                'schema': {
                    'type': 'string'
                },
            },
            {
                'name': 'a',
                'description': '积分区间的左端点，例如1.0',
                'required': True,
                'schema': {
                    'type': 'string'
                },
            },
            {
                'name': 'b',
                'description': '积分区间的右端点，例如5.0',
                'required': True,
                'schema': {
                    'type': 'string'
                },
            },
        ],
    },
    {
        "name": "calculate_sqrt",
        "description": "计算一个数值的平方根。",
        "parameters": [
            {
                'name': 'y',
                'description': '被开方数',
                'required': True,
                'schema': {
                    'type': 'string'
                },
            },
        ],
    },
]

messages = [{"role": "user", "content": "函数f(x)=x**2在区间[0,5]上的定积分是多少？"}]
response = openai.ChatCompletion.create(
    model="qwen",
    messages=messages,
    temperature=0,
    functions=functions,
    stop=["Observation:"],
)
print("Function Call results:")
print(response.choices[0].message.function_call)

response = openai.ChatCompletion.create(
    model="qwen",
    messages=messages,
    temperature=0,
    functions=functions,
    stop=["Observation:"],
    stream=True,
)
print("Function Call streaming results:")
for r in response:
    function_call = r.choices[0].delta.get("function_call", None)
    if function_call:
        print(function_call.arguments)
