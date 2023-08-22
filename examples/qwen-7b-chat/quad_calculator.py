import json
import math

import openai
from loguru import logger
from scipy import integrate


def calculate_quad(formula_str: str, a: float, b: float) -> float:
    """ 计算数值积分 """
    return integrate.quad(eval('lambda x: ' + formula_str), a, b)[0]


def calculate_sqrt(y: float) -> float:
    """ 计算平方根 """
    return math.sqrt(y)


class QuadCalculator:
    def __init__(self, openai_api_base, openai_api_key="xxx"):
        openai.api_base = openai_api_base
        openai.api_key = openai_api_key

        self.functions = [
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

    def run(self, query: str) -> str:
        # Step 1: send the conversation and available functions to model
        messages = [{"role": "user", "content": query}]
        response = openai.ChatCompletion.create(
            model="qwen",
            messages=messages,
            temperature=0,
            functions=self.functions,
            stop=["Observation:"]
        )

        while True:
            if response["choices"][0]["finish_reason"] == "stop":
                answer = response["choices"][0]["message"]["content"]
                return answer

            elif response["choices"][0]["finish_reason"] == "function_call":
                response_message = response["choices"][0]["message"]
                # Step 2: check if model wanted to call a function
                if response_message.get("function_call"):
                    logger.info(f"Function call: {response_message['function_call']}")
                    # Step 3: call the function
                    # Note: the JSON response may not always be valid; be sure to handle errors
                    available_functions = {
                        "calculate_quad": calculate_quad,
                        "calculate_sqrt": calculate_sqrt,
                    }

                    function_name = response_message["function_call"]["name"]
                    fuction_to_call = available_functions[function_name]
                    function_args = json.loads(response_message["function_call"]["arguments"])
                    logger.info(f"Function args: {function_args}")

                    for k in ["a", "b", "y"]:
                        if k in function_args:
                            function_args[k] = float(function_args[k])
                    function_response = fuction_to_call(**function_args)
                    logger.info(f"Function response: {function_response}")

                    # Step 4: send the info on the function call and function response to model
                    messages.append(response_message)  # extend conversation with assistant's reply
                    messages.append(
                        {
                            "role": "function",
                            "name": function_name,
                            "content": function_response,
                        }
                    )  # extend conversation with function response

                    response = openai.ChatCompletion.create(
                        model="qwen",
                        messages=messages,
                        temperature=0,
                        stop=["Observation:"],
                    )  # get a new response from model where it can see the function response


if __name__ == '__main__':
    openai_api_base = "http://192.168.0.53:7891/v1"
    query = "函数f(x)=x**2在区间[0,5]上的定积分是多少？其平方根又是多少？"

    calculator = QuadCalculator(openai_api_base)
    answer = calculator.run(query)
    print(answer)
