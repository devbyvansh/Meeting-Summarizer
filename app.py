import os
import tempfile

import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openrouter import ChatOpenRouter
from openrouter import OpenRouter


SUMMARY_MODEL = "nvidia/nemotron-3-super-120b-a12b"
ALLOWED_AUDIO_TYPES = ["wav", "mp3", "m4a"]


st.set_page_config(page_title="MeetingPilot", page_icon=":material/insights:", layout="wide")


def initialize_state() -> None:
    st.session_state.setdefault("transcript", "")
    st.session_state.setdefault("summary", "")
    st.session_state.setdefault("topics", "")
    st.session_state.setdefault("sentiment", "")
    st.session_state.setdefault("answer", "")
    st.session_state.setdefault("uploaded_signature", None)


def build_client(api_key: str) -> OpenRouter:
    return OpenRouter(api_key=api_key)


def audio_to_text(client: OpenRouter, audio_path: str) -> str:
    with open(audio_path, "rb") as audio_file:
        transcript = client.stt.create_transcription_multipart(
            file={
                "file_name": os.path.basename(audio_path),
                "content": audio_file,
            },
            model="openai/whisper-1",
        )
    return transcript.text


def summarize_meeting(client: OpenRouter, meeting_transcript: str) -> str:
    chat = ChatOpenRouter(
        client=client,
        model=SUMMARY_MODEL,
        temperature=0.3,
    )
    messages = [
        SystemMessage(
            content="""
You are an expert AI Meeting Assistant.

Generate professional meeting notes.

Return:

# Meeting Summary

## Overview

## Objectives

## Key Discussion Points

## Decisions Made

## Action Items

| Task | Owner | Deadline |

If information is missing write "Not Mentioned".

## Risks / Blockers

## Important Dates

## Next Steps

## Conclusion

Do not hallucinate.
"""
        ),
        HumanMessage(content=meeting_transcript),
    ]
    response = chat.invoke(messages)
    return response.content


def question_answer(client: OpenRouter, summary: str, question: str) -> str:
    chat = ChatOpenRouter(
        client=client,
        model=SUMMARY_MODEL,
        temperature=0.3,
    )
    messages = [
        SystemMessage(
            "You are an expert AI Meeting Assistant. Answer only from the meeting summary and do not invent facts."
        ),
        HumanMessage(
            content=f"""Meeting Summary:

{summary}

Question:
{question}""",
        ),
    ]
    response = chat.invoke(messages)
    return response.content


def topic_detection(client: OpenRouter, summary: str) -> str:
    chat = ChatOpenRouter(
        client=client,
        model=SUMMARY_MODEL,
        temperature=0.3,
    )
    messages = [
        SystemMessage(
            content="""
You are an expert AI Meeting Assistant.

Your task is to identify the main topics discussed in the meeting.

Rules:
- Return only the major topics.
- Merge similar topics.
- Do not explain each topic.
- Return between 3 and 10 topics.
- Output as a bullet list.
"""
        ),
        HumanMessage(
            content=f"""Meeting Summary:

{summary}

Extract the discussion topics.""",
        ),
    ]
    response = chat.invoke(messages)
    return response.content


def sentiment_analysis(client: OpenRouter, summary: str) -> str:
    chat = ChatOpenRouter(
        client=client,
        model=SUMMARY_MODEL,
        temperature=0.3,
    )
    messages = [
        SystemMessage(
            content="""
You are an expert AI Meeting Assistant.

Your task is to analyze the sentiment of the meeting summary.

Rules:
- Return only the overall sentiment (positive, negative, or neutral).
- Do not provide any explanation.
"""
        ),
        HumanMessage(
            content=f"""Meeting Summary:

{summary}

Analyze the sentiment.""",
        ),
    ]
    response = chat.invoke(messages)
    return response.content


def save_uploaded_audio(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        return temp_file.name


initialize_state()

st.title("MeetingPilot")
st.subheader("Upload a meeting recording, keep your API key private, and generate notes in one place.")

with st.sidebar:
    st.header("Setup")
    api_key = st.text_input(
        "OpenRouter API key",
        type="password",
        placeholder="Enter your OpenRouter API key",
    )
    st.caption("The key stays hidden in the app and is not hard-coded.")
    st.divider()
    st.write("Model")
    st.code(SUMMARY_MODEL, language="text")

st.info("Upload a .wav, .mp3, or .m4a meeting file, then generate a transcript and summary.")

uploaded_audio = st.file_uploader(
    "Audio file",
    type=ALLOWED_AUDIO_TYPES,
    help="Supported formats: WAV, MP3, and M4A.",
)

if uploaded_audio is not None:
    signature = (uploaded_audio.name, uploaded_audio.size)
    if st.session_state.uploaded_signature != signature:
        st.session_state.uploaded_signature = signature
        st.session_state.transcript = ""
        st.session_state.summary = ""
        st.session_state.topics = ""
        st.session_state.sentiment = ""
        st.session_state.answer = ""

    st.audio(uploaded_audio.getvalue())

process_columns = st.columns(3)

with process_columns[0]:
    generate_transcript = st.button("Generate transcript", type="primary", width="stretch")

with process_columns[1]:
    generate_summary = st.button("Generate summary", width="stretch")

with process_columns[2]:
    generate_analysis = st.button("Generate topics and sentiment", width="stretch")

if not api_key:
    st.warning("Enter your OpenRouter API key in the sidebar to enable transcription and analysis.")

if generate_transcript:
    if not api_key:
        st.error("Add your OpenRouter API key first.")
    elif uploaded_audio is None:
        st.error("Upload an audio file first.")
    else:
        client = build_client(api_key)
        with st.spinner("Transcribing audio with Whisper..."):
            temp_audio_path = save_uploaded_audio(uploaded_audio)
            st.session_state.transcript = audio_to_text(client, temp_audio_path)
        st.success("Transcript generated.")

if generate_summary:
    if not api_key:
        st.error("Add your OpenRouter API key first.")
    elif not st.session_state.transcript:
        st.error("Generate a transcript before creating a summary.")
    else:
        client = build_client(api_key)
        with st.spinner("Writing meeting summary..."):
            st.session_state.summary = summarize_meeting(client, st.session_state.transcript)
        st.success("Summary generated.")

if generate_analysis:
    if not api_key:
        st.error("Add your OpenRouter API key first.")
    elif not st.session_state.summary:
        st.error("Generate a summary before extracting topics and sentiment.")
    else:
        client = build_client(api_key)
        with st.spinner("Analyzing topics and sentiment..."):
            st.session_state.topics = topic_detection(client, st.session_state.summary)
            st.session_state.sentiment = sentiment_analysis(client, st.session_state.summary)
        st.success("Analysis generated.")

left_panel, right_panel = st.columns(2)

with left_panel:
    st.subheader("Transcript")
    if st.session_state.transcript:
        st.text_area("Meeting transcript", st.session_state.transcript, height=320)
    else:
        st.caption("Transcript will appear here after processing.")

    st.subheader("Ask a question")
    question = st.text_input("Question", placeholder="What were the main decisions?")
    ask_button = st.button("Ask about the summary", width="stretch")

    if ask_button:
        if not api_key:
            st.error("Add your OpenRouter API key first.")
        elif not st.session_state.summary:
            st.error("Generate a summary before asking questions.")
        elif not question.strip():
            st.error("Enter a question first.")
        else:
            client = build_client(api_key)
            with st.spinner("Answering your question..."):
                st.session_state.answer = question_answer(client, st.session_state.summary, question.strip())

    if st.session_state.answer:
        st.markdown("#### Answer")
        st.write(st.session_state.answer)

with right_panel:
    st.subheader("Summary")
    if st.session_state.summary:
        st.markdown(st.session_state.summary)
    else:
        st.caption("Summary will appear here after processing.")

    st.subheader("Topics")
    if st.session_state.topics:
        st.write(st.session_state.topics)
    else:
        st.caption("Topic extraction will appear here after analysis.")

    st.subheader("Sentiment")
    if st.session_state.sentiment:
        st.write(st.session_state.sentiment)
    else:
        st.caption("Sentiment will appear here after analysis.")
