import json

import openai

from tools import send_email, send_email_action, SkillFunctions

openai.api_base = "http://192.168.0.53:7891/v1"
openai.api_key = "xxx"


def run_conversation():
    # Step 1: send the conversation and available functions to model
    messages = [{"role": "user", "content": "给小王发个邮件，告诉他我晚饭不回家吃了"}]
    functions = [
        {
            "name_for_human":
                "邮件助手",
            "name_for_model":
                SkillFunctions.SendEmail.value,
            "description_for_model":
                "邮件助手是一个可以帮助用户发送邮件的工具。",
            "parameters": [
                {
                    'name': 'receiver',
                    'description': '邮件接收者',
                    'required': True,
                    'schema': {
                        'type': 'string'
                    },
                },
                {
                    'content': 'content',
                    'description': '邮件内容',
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
            SkillFunctions.SendEmail.value: send_email_action,
        }  # only one function in this example
        function_name = response_message["function_call"]["name"]
        fuction_to_call = available_functions[function_name]
        function_args = json.loads(response_message["function_call"]["arguments"])

        email_info = send_email(
            receiver=function_args.get("receiver"),
            content=function_args.get("content")
        )
        fuction_to_call(**email_info)
        print("邮件已发送")


if __name__ == '__main__':
    run_conversation()
