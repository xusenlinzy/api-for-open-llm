import logging
import os
import shutil

import gradio as gr
import openai
from backoff import on_exception, expo

from tools.doc_qa import DocQAPromptAdapter
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


doc_adapter = DocQAPromptAdapter()


def add_llm(model_name, api_base, models):
    models = models or {}
    models.update(
        {
            model_name: api_base
        }
    )
    choices = [m[0] for m in models.items()]
    return "",  "", models, gr.Dropdown.update(choices=choices, value=choices[0])


def set_openai_env(api_base):
    openai.api_base = api_base
    doc_adapter.embeddings.openai_api_base = api_base


def get_file_list():
    if not os.path.exists("doc_store"):
        return []
    return os.listdir("doc_store")


file_list = get_file_list()


def upload_file(file):
    if not os.path.exists("doc_store"):
        os.mkdir("docs")

    if file is not None:
        filename = os.path.basename(file.name)
        shutil.move(file.name, f"doc_store/{filename}")
        file_list.insert(0, filename)
        return gr.Dropdown.update(choices=file_list, value=filename)


def add_vector_store(filename, model_name, models):
    api_base = models[model_name]
    set_openai_env(api_base)
    if filename is not None:
        vs_path = f"vector_store/{filename.split('.')[0]}-{filename.split('.')[-1]}"
        if not os.path.exists(vs_path):
            doc_adapter.create_vector_store(f"doc_store/{filename}", vs_path=vs_path)
            msg = f"Successfully added vector store for {filename}!"
        else:
            doc_adapter.reset_vector_store(vs_path=vs_path)
            msg = f"Successfully loaded vector store for {filename}!"
    else:
        msg = "Please select a file!"
    return msg


@on_exception(expo, openai.error.RateLimitError, max_tries=5)
def chat_completions_create(params):
    return openai.ChatCompletion.create(**params)


def predict(
    model_name,
    models,
    text,
    chatbot,
    history,
    top_p,
    temperature,
    max_tokens,
    memory_k,
    pattern
):
    api_base = models[model_name]
    set_openai_env(api_base)

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
            "content": doc_adapter(text) if pattern != "通用" else text
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
    models,
    text,
    chatbot,
    history,
    top_p,
    temperature,
    max_tokens,
    memory_k,
    pattern
):
    logging.info("Retry...")
    if len(history) == 0:
        yield chatbot, history, "Empty context."
        return
    chatbot.pop()
    inputs = history.pop()[0]
    for x in predict(
        model_name,
        models,
        inputs,
        chatbot,
        history,
        top_p,
        temperature,
        max_tokens,
        memory_k,
        pattern
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
            with gr.Row():
                chatbot = gr.Chatbot(elem_id="chuanhu_chatbot").style(height="100%")
            with gr.Row():
                with gr.Column(scale=12):
                    user_input = gr.Textbox(
                        show_label=False, placeholder="Enter text"
                    ).style(container=False)
                with gr.Column(min_width=70, scale=1):
                    submitBtn = gr.Button("发送")
                with gr.Column(min_width=70, scale=1):
                    cancelBtn = gr.Button("停止")
            with gr.Row():
                emptyBtn = gr.Button(
                    "🧹 新的对话",
                )
                retryBtn = gr.Button("🔄 重新生成")
                delLastBtn = gr.Button("🗑️ 删除最旧对话")
        with gr.Column():
            with gr.Column(min_width=50, scale=1):
                with gr.Tab(label="模型"):
                    model_name = gr.Textbox(
                        placeholder="chatglm",
                        label="模型名称",
                    )
                    api_base = gr.Textbox(
                        placeholder="https://0.0.0.0:80/v1",
                        label="模型接口地址",
                    )
                    add_model = gr.Button("添加模型")
                    with gr.Accordion(open=False, label="所有模型配置"):
                        models = gr.Json()
                    select_model = gr.Dropdown(
                        choices=[m[0] for m in models.value.items()] if models.value else [],
                        value=[m[0] for m in models.value.items()][0] if models.value else None,
                        label="选择模型"
                    )
                with gr.Tab(label="本地知识问答"):
                    pattern = gr.Radio(
                        choices=['通用', '知识库'],
                        label="问答模式",
                        value='通用',
                        interactive=True,
                    )
                    select_file = gr.Dropdown(
                        choices=file_list,
                        label="选择文件",
                        interactive=True,
                        value=file_list[0] if len(file_list) > 0 else None
                    )
                    file = gr.File(
                        label="上传文件",
                        visible=True,
                        file_types=['.txt', '.md', '.docx', '.pdf']
                    )
                    add = gr.Button(value="添加到知识库")
                with gr.Tab(label="参数"):
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

    add_model.click(
        add_llm,
        inputs=[model_name, api_base, models],
        outputs=[model_name, api_base, models, select_model],
    )

    file.upload(
        upload_file,
        inputs=file,
        outputs=select_file,
    )

    add.click(
        add_vector_store,
        inputs=[select_file, select_model, models],
        outputs=status_display,
    )

    predict_args = dict(
        fn=predict,
        inputs=[
            select_model,
            models,
            user_question,
            chatbot,
            history,
            top_p,
            temperature,
            max_tokens,
            memory_k,
            pattern
        ],
        outputs=[chatbot, history, status_display],
        show_progress=True,
    )
    retry_args = dict(
        fn=retry,
        inputs=[
            select_model,
            models,
            user_question,
            chatbot,
            history,
            top_p,
            temperature,
            max_tokens,
            memory_k,
            pattern
        ],
        outputs=[chatbot, history, status_display],
        show_progress=True,
    )

    reset_args = dict(fn=reset_textbox, inputs=[], outputs=[user_input, status_display])

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
