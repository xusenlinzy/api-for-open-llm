import json

import openai
from loguru import logger
from scipy import integrate


def calculate_quad(formula_str: str, a: float, b: float) -> float:
    """ 数值积分 """
    return integrate.quad(eval('lambda x: ' + formula_str), a, b)[0]


class QuadCalculator:
    def __init__(self, openai_api_base, openai_api_key="xxx"):
        openai.api_base = openai_api_base
        openai.api_key = openai_api_key

    def run(self, query):
        # Step 1: send the conversation and available functions to model
        messages = [{"role": "user", "content": query}]
        functions = [
            {
                "name_for_human":
                    "定积分计算器",
                "name_for_model":
                    "calculate_quad",
                "description_for_model":
                    "定积分计算器是一个可以计算给定区间内函数定积分数值的工具。",
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
            }
        ]
        response = openai.ChatCompletion.create(
            model="qwen",
            messages=messages,
            temperature=0,
            functions=functions,
            stop=["Observation:"]
        )

        answer = ""
        response_message = response["choices"][0]["message"]
        # Step 2: check if model wanted to call a function
        if response_message.get("function_call"):
            # Step 3: call the function
            # Note: the JSON response may not always be valid; be sure to handle errors
            available_functions = {
                "calculate_quad": calculate_quad,
            }  # only one function in this example
            function_name = response_message["function_call"]["name"]
            fuction_to_call = available_functions[function_name]
            function_args = json.loads(response_message["function_call"]["arguments"])
            logger.info(f"Function args: {function_args}")

            function_args["a"] = float(function_args["a"])
            function_args["b"] = float(function_args["b"])
            function_response = fuction_to_call(**function_args)

            # Step 4: send the info on the function call and function response to model
            messages.append(response_message)  # extend conversation with assistant's reply
            messages.append(
                {
                    "role": "function",
                    "content": function_response,
                }
            )  # extend conversation with function response

            second_response = openai.ChatCompletion.create(
                model="qwen",
                messages=messages,
                temperature=0,
                functions=functions,
            )  # get a new response from model where it can see the function response
            answer = second_response["choices"][0]["message"]["content"]

        return answer[answer.index("Final Answer:") + 14:] if answer else answer


if __name__ == '__main__':
    openai_api_base = "http://192.168.0.53:7891/v1"
    query = "函数f(x)=x**2在区间[0,5]上的定积分是多少？"

    calculator = QuadCalculator(openai_api_base)
    answer = calculator.run(query)
    print(answer)
