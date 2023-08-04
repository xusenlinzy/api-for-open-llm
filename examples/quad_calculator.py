import json

import openai

from tools import calculate_quad, SkillFunctions

openai.api_base = "http://192.168.0.53:7891/v1"
openai.api_key = "xxx"


def run_conversation():
    # Step 1: send the conversation and available functions to model
    messages = [{"role": "user", "content": "函数f(x)=x**2在区间[0,5]上的定积分是多少？"}]
    functions = [
        {
            "name_for_human":
                "定积分计算器",
            "name_for_model":
                SkillFunctions.Quad.value,
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
    print(response)

    response_message = response["choices"][0]["message"]
    # Step 2: check if model wanted to call a function
    if response_message.get("function_call"):
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            SkillFunctions.Quad.value: calculate_quad,
        }  # only one function in this example
        function_name = response_message["function_call"]["name"]
        fuction_to_call = available_functions[function_name]
        function_args = json.loads(response_message["function_call"]["arguments"])

        print(function_args)
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
        print(second_response["choices"][0]["message"]["content"])


if __name__ == '__main__':
    run_conversation()
