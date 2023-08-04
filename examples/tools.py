import smtplib
from email.header import Header
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum


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
