import openai

openai.api_key = "xxx"
openai.api_base = "http://45.63.96.171:8080/v1"


def translate(query):
    d = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": f"请将下面的句子准确翻译成流利通顺的中文。\n\n{query}"
            }
        ]
    )
    return d.choices[0].message.content


print(translate("How many triple type bonds are there?"))
