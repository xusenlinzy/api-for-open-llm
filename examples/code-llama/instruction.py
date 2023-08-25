""" https://github.com/facebookresearch/codellama/blob/main/example_instructions.py """

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

llm = ChatOpenAI(
    model_name="code-llama",
    openai_api_base="http://192.168.0.53:7891/v1",
    openai_api_key="xxx",
)


def test():
    instructions = [
        [
            HumanMessage(content="In Bash, how do I list all text files in the current directory (excluding subdirectories) that have been modified in the last month?")
        ],
        [
            HumanMessage(content="What is the difference between inorder and preorder traversal? Give an example in Python.")
        ],
        [
            SystemMessage(content="Provide answers in JavaScript"),
            HumanMessage(content="Write a function that computes the set of sums of all contiguous sublists of a given list.")
        ],
    ]

    for instruction in instructions:
        result = llm(instruction)
        for msg in instruction:
            print(f"{msg.type.capitalize()}: {msg.content}\n")

        print(
            f"> AI: {result.content}"
        )
        print("\n==================================\n")


if __name__ == "__main__":
    test()
