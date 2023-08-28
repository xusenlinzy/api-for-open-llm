import openai
import pandas as pd
import streamlit as st

st.set_page_config(page_title="LLM Finetuning Web UI", page_icon=":smiley:", layout="wide")

with st.sidebar:
    api_key = st.text_input('Enter your API key:', '')

# If api_key is entered, read the contents and process the data
api_key = api_key or "xxx"

openai.api_base = "http://localhost:8000/v1"
openai.api_key = api_key
st.title("ChatGPT Finetuning WebUI")

st.subheader("Files")
files = openai.File.list()
st.table(pd.DataFrame(sorted(files.data, key=lambda k: -k['created_at'])))

st.subheader("Jobs")
jobs = openai.FineTuningJob.list()
st.table(pd.DataFrame(sorted(jobs.data, key=lambda k: -k['created_at'])))

st.subheader("Finetuned Models")
models = openai.Model.list()
st.table(pd.DataFrame([d for d in models.data if d["id"].startswith("ft")]))

st.subheader("Debug Info")
response_display = st.empty()

with st.sidebar:
    file = st.file_uploader("Upload a file", accept_multiple_files=False)

    file_ids = [d["id"] for d in sorted(files.data, key=lambda k: -k['created_at'])]
    file_id = st.selectbox("Select a file", file_ids)

    job_ids = [d["id"] for d in sorted(jobs.data, key=lambda k: -k['created_at'])]
    job_id = st.selectbox("Select a job", job_ids)

    if file:
        uploaded_file = openai.File.create(file=file, purpose='fine-tune', user_provided_filename=file.name)
        response_display.write(uploaded_file)

    if st.button("Delete File") and file_id:
        deleted_file = openai.File.delete(file_id)
        response_display.write(deleted_file)

    if st.button("Create Fine-Tuning Job") and file_id:
        job = openai.FineTuningJob.create(training_file=file_id, model='gpt-3.5-turbo')
        response_display.write(job)

    if st.button("Get Fine-Tuning Job Detail") and job_id:
        job = openai.FineTuningJob.retrieve(job_id)
        response_display.write(job)

    if st.button("List Job Events") and job_id:
        events = openai.FineTuningJob.list_events(id=job_id, limit=10)
        for event in events.data:
            response_display.write(event)

    if st.button("Cancel Job") and job_id:
        cancelled_job = openai.FineTuningJob.cancel(job_id)
        response_display.write(cancelled_job)
