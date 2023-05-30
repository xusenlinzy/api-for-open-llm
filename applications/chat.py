import argparse
import os

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

from tools.memory import CustomConversationBufferWindowMemory
from tools.prompt import ChatPromptTEMPLATE


def main():
    llm = OpenAI(
        model_name=args.model_name,
        streaming=True,
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )

    memory = CustomConversationBufferWindowMemory.create(args.model_name, k=args.k)
    chat_chain = ConversationChain(llm=llm, memory=memory, verbose=False)
    chat_chain.prompt = PromptTemplate(
        input_variables=["history", "input"], template=ChatPromptTEMPLATE.create(args.model_name)
    )

    while True:
        user_input = input("HUMAN: ")
        print("AI: ")
        chat_chain.predict(input=user_input)
        print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat server.")
    parser.add_argument(
        '--api_base', type=str, help='model api base', required=True,
    )
    parser.add_argument(
        '--model_name', type=str, help='chatglm, moss, phoenix, chinese-llama-alpaca', default='chatglm'
    )
    parser.add_argument(
        '--k', type=int, help='max memory length', default=5
    )
    args = parser.parse_args()

    os.environ["OPENAI_API_BASE"] = args.api_base
    os.environ["OPENAI_API_KEY"] = "xxx"

    main()
