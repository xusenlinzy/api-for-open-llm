""" https://github.com/facebookresearch/codellama/blob/main/example_completion.py """

from langchain.llms import OpenAI

llm = OpenAI(
    model_name="code-llama",
    openai_api_base="http://192.168.0.53:7891/v1",
    openai_api_key="xxx",
)


def test():
    # For these prompts, the expected answer is the natural continuation of the prompt
    prompts = [
        """\
import socket

def ping_exponential_backoff(host: str):""",
        """\
import argparse

def main(string: str):
    print(string)
    print(string[::-1])

if __name__ == "__main__":"""
    ]

    for prompt in prompts:
        result = llm(prompt)
        print(prompt)
        print(f"> {result}")
        print("\n==================================\n")


if __name__ == "__main__":
    test()
