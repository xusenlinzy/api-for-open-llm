import json
import smtplib
from email.header import Header
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import openai
from loguru import logger


def send_email_action(receiver: str, content: str):
    """ 发送邮件操作 """
    if not receiver:
        return

    # 邮件配置
    smtp_server = "smtp.163.com"
    smtp_port = 25
    sender_email = "xxx@163.com"  # 发件人邮箱地址
    receiver_email = receiver  # 收件人邮箱地址
    password = '***'  # SMTP授权密码

    # 构建邮件内容
    message = MIMEMultipart()
    message["From"] = Header('AI <%s>' % sender_email)
    message["To"] = receiver_email
    message["Subject"] = "我是您的AI助理，您有一封邮件请查看"

    body = content
    message.attach(MIMEText(body, "plain"))

    # 连接到邮件服务器并发送邮件
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message.as_string())


def send_email(receiver: str, content: str = "") -> dict:
    """ 供Function Calling使用的输出处理函数 """
    Contact = {"小王": "1659821119@qq.com"}  # 通讯录
    email_info = {
        "receiver": Contact[receiver],
        "content": content
    }
    return email_info


class EmailSender:
    def __init__(self, openai_api_base, openai_api_key="xxx"):
        openai.api_base = openai_api_base
        openai.api_key = openai_api_key

    def run(self, query):
        # Step 1: send the conversation and available functions to model
        messages = [{"role": "user", "content": query}]
        functions = [
            {
                "name": "send_email",
                "description": "邮件助手是一个可以帮助用户发送邮件的工具。",
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
                        'name': 'content',
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

        response_message = response["choices"][0]["message"]
        # Step 2: check if model wanted to call a function
        if response_message.get("function_call"):
            logger.info(f"Function call: {response_message['function_call']}")
            # Step 3: call the function
            # Note: the JSON response may not always be valid; be sure to handle errors
            available_functions = {
                "send_email": send_email_action,
            }  # only one function in this example
            function_name = response_message["function_call"]["name"]
            fuction_to_call = available_functions[function_name]
            function_args = json.loads(response_message["function_call"]["arguments"])
            logger.info(f"Function args: {function_args}")

            email_info = send_email(
                receiver=function_args.get("receiver"),
                content=function_args.get("content")
            )
            fuction_to_call(**email_info)
            logger.info("邮件已发送")


if __name__ == '__main__':
    openai_api_base = "http://192.168.0.53:7891/v1"
    query = "给小王发个邮件，告诉他今天晚上一起吃个饭"
    sender = EmailSender(openai_api_base)
    sender.run(query)
