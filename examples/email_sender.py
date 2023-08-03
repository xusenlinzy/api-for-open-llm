import json
import smtplib
from email.header import Header
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum

import openai

from react_prompt import get_qwen_reat_prompt

openai.api_base = "http://192.168.0.53:7891/v1"
openai.api_key = "xxx"


class SkillFunctions(Enum):

    SendEmail = "send_email"


def send_email_action(receiver: str, content: str):
    """ 发送邮件操作 """
    if not receiver:
        return

    # 邮件配置
    smtp_server = "smtp.163.com"
    smtp_port = 25
    sender_email = "xxx@163.com"  # 发件人邮箱地址
    receiver_email = receiver  # 收件人邮箱地址
    password = 'YRXGBDYXXFVKJBUO'  # SMTP授权密码

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


#
def send_email(receiver: str, content: str = "") -> dict:
    """ 供Function Calling使用的输出处理函数 """
    Contact = {"小王": "1659821119@qq.com"}  # 通讯录
    email_info = {
        "receiver": Contact[receiver],
        "content": content
    }
    return email_info


def main():
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
    messages = [{"role": "user", "content": "给小王发个邮件，告诉他我晚饭不回家吃了"}]
    messages = get_qwen_reat_prompt(functions, messages)
    response = openai.ChatCompletion.create(
        model="qwen",
        messages=messages,
        temperature=0,
        stop=["Observation:"]
    )

    content = response["choices"][0]["message"]["content"].strip()
    print(content)
    if "Action Input:" in content:
        arguments = json.loads(content[content.index("Action Input:") + 14:])
        email_info = send_email(
            receiver=arguments.get('receiver'),
            content=arguments.get('content')
        )
        print(email_info)
        send_email_action(**email_info)
        print('邮件已发送')


if __name__ == '__main__':
    main()
