import logging

import gradio as gr
import openai
from backoff import on_exception, expo

from tools.web.overwrites import postprocess, reload_javascript
from tools.web.presets import (
    small_and_beautiful_theme,
    title,
    description,
    description_top,
    CONCURRENT_COUNT
)
from tools.web.utils import (
    convert_to_markdown,
    shared_state,
    reset_textbox,
    cancel_outputing,
    transfer_input,
    reset_state,
    delete_last_conversation
)

openai.api_key = "xxx"
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
)


@on_exception(expo, openai.error.RateLimitError, max_tries=5)
def chat_completions_create(params):
    return openai.ChatCompletion.create(**params)


def predict(
    model_name,
    api_base,
    text,
    chatbot,
    history,
    top_p,
    temperature,
    max_tokens,
    memory_k
):
    if text == "":
        yield chatbot, history, "Empty context."
        return

    if history is None:
        history = []

    messages = []
    for h in history[-memory_k:]:
        messages.extend(
            [
                {
                    "role": "user",
                    "content": h[0]
                },
                {
                    "role": "assistant",
                    "content": h[1]
                }
            ]
        )
    messages.append(
        {
            "role": "user",
            "content": text
        }
    )

    params = dict(
        stream=True,
        messages=messages,
        model=model_name,
        top_p=top_p,
        temperature=temperature,
        max_tokens=max_tokens
    )

    openai.api_base = api_base
    res = chat_completions_create(params)
    x = ""
    for openai_object in res:
        delta = openai_object.choices[0]["delta"]
        if "content" in delta:
            x += delta["content"]
        a, b = [[y[0], convert_to_markdown(y[1])] for y in history] + [
            [text, convert_to_markdown(x)]
        ], history + [[text, x]]
        yield a, b, "Generating..."

    if shared_state.interrupted:
        shared_state.recover()
        try:
            yield a, b, "Stop: Success"
            return
        except:
            pass

    try:
        yield a, b, "Generate: Success"
    except:
        pass


def retry(
    model_name,
    api_base,
    text,
    chatbot,
    history,
    top_p,
    temperature,
    max_tokens,
    memory_k,
):
    logging.info("Retry...")
    if len(history) == 0:
        yield chatbot, history, "Empty context."
        return
    chatbot.pop()
    inputs = history.pop()[0]
    for x in predict(
        model_name,
        api_base,
        inputs,
        chatbot,
        history,
        top_p,
        temperature,
        max_tokens,
        memory_k
    ):
        yield x


gr.Chatbot.postprocess = postprocess

with open("assets/custom.css", "r", encoding="utf-8") as f:
    customCSS = f.read()

with gr.Blocks(css=customCSS, theme=small_and_beautiful_theme) as demo:
    history = gr.State([])
    user_question = gr.State("")
    with gr.Row():
        gr.HTML(title)
        status_display = gr.Markdown("Success", elem_id="status_display")
    gr.Markdown(description_top)
    with gr.Row().style(equal_height=True, scale=1):
        with gr.Column(scale=5):
            with gr.Row(scale=1):
                chatbot = gr.Chatbot(elem_id="chuanhu_chatbot").style(height="100%")
            with gr.Row(scale=1):
                with gr.Column(scale=12):
                    user_input = gr.Textbox(
                        show_label=False, placeholder="Enter text"
                    ).style(container=False)
                with gr.Column(min_width=70, scale=1):
                    submitBtn = gr.Button("Send")
                with gr.Column(min_width=70, scale=1):
                    cancelBtn = gr.Button("Stop")

            with gr.Row(scale=1):
                emptyBtn = gr.Button(
                    "🧹 New Conversation",
                )
                retryBtn = gr.Button("🔄 Regenerate")
                delLastBtn = gr.Button("🗑️ Remove Last Turn")
        with gr.Column():
            with gr.Column(min_width=50, scale=1):
                api_base = gr.Textbox(
                    placeholder="https://0.0.0.0:80/v1",
                    label="API base",
                )
                model_name = gr.Dropdown(
                    choices=["chatglm", "moss", "phoenix"],
                    value="chatglm",
                    label="Model name"
                )
                top_p = gr.Slider(
                    minimum=-0,
                    maximum=1.0,
                    value=0.95,
                    step=0.05,
                    interactive=True,
                    label="Top-p",
                )
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=1,
                    step=0.1,
                    interactive=True,
                    label="Temperature",
                )
                max_tokens = gr.Slider(
                    minimum=0,
                    maximum=512,
                    value=512,
                    step=8,
                    interactive=True,
                    label="Max Generation Tokens",
                )
                memory_k = gr.Slider(
                    minimum=0,
                    maximum=10,
                    value=5,
                    step=1,
                    interactive=True,
                    label="Max Memory Window Size",
                )

    gr.Markdown(description)

    predict_args = dict(
        fn=predict,
        inputs=[
            model_name,
            api_base,
            user_question,
            chatbot,
            history,
            top_p,
            temperature,
            max_tokens,
            memory_k
        ],
        outputs=[chatbot, history, status_display],
        show_progress=True,
    )
    retry_args = dict(
        fn=retry,
        inputs=[
            model_name,
            api_base,
            user_question,
            chatbot,
            history,
            top_p,
            temperature,
            max_tokens,
            memory_k
        ],
        outputs=[chatbot, history, status_display],
        show_progress=True,
    )

    reset_args = dict(fn=reset_textbox, inputs=[], outputs=[user_input, status_display])

    # Chatbot
    cancelBtn.click(cancel_outputing, [], [status_display])
    transfer_input_args = dict(
        fn=transfer_input,
        inputs=[user_input],
        outputs=[user_question, user_input, submitBtn, cancelBtn],
        show_progress=True,
    )

    user_input.submit(**transfer_input_args).then(**predict_args)

    submitBtn.click(**transfer_input_args).then(**predict_args)

    emptyBtn.click(
        reset_state,
        outputs=[chatbot, history, status_display],
        show_progress=True,
    )
    emptyBtn.click(**reset_args)

    retryBtn.click(**retry_args)

    delLastBtn.click(
        delete_last_conversation,
        [chatbot, history],
        [chatbot, history, status_display],
        show_progress=True,
    )

demo.title = "OpenLLM Chatbot 🚀 "

if __name__ == "__main__":
    reload_javascript()
    demo.queue(concurrency_count=CONCURRENT_COUNT).launch(server_name="0.0.0.0")
