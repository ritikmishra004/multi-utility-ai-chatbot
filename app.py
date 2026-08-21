import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langraph_rag_backend import (
    chatbot,
    ingest_pdf,
    retrieve_all_threads,
    thread_document_metadata,
)


st.set_page_config(
    page_title="Chatbot",
    page_icon="🤖",
    layout="wide",
)


def generate_thread_id():
    return str(uuid.uuid4())


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def reset_chat():
    thread_id = generate_thread_id()

    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = []

    add_thread(thread_id)


def load_conversation(thread_id):
    state = chatbot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    return state.values.get("messages", [])


if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}


add_thread(st.session_state["thread_id"])

thread_key = st.session_state["thread_id"]

thread_docs = st.session_state["ingested_docs"].setdefault(
    thread_key,
    {}
)

threads = st.session_state["chat_threads"][::-1]

selected_thread = None


st.sidebar.title("Chatbot")


if st.sidebar.button(
    "➕ New Chat",
    use_container_width=True,
):
    reset_chat()
    st.rerun()


if thread_docs:
    latest_doc = list(thread_docs.values())[-1]

    st.sidebar.success(
        f"📄 {latest_doc.get('filename')}\n\n"
        f"{latest_doc.get('chunks')} chunks · "
        f"{latest_doc.get('documents')} pages"
    )
else:
    st.sidebar.info("No PDF indexed yet.")


uploaded_pdf = st.sidebar.file_uploader(
    "Upload a PDF",
    type=["pdf"],
)


if uploaded_pdf:

    if uploaded_pdf.name in thread_docs:

        st.sidebar.info(
            f"`{uploaded_pdf.name}` is already indexed "
            "for this chat."
        )

    else:

        with st.sidebar.status(
            "Indexing PDF...",
            expanded=True,
        ) as status_box:

            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )

            thread_docs[uploaded_pdf.name] = summary

            status_box.update(
                label="✅ PDF indexed",
                state="complete",
                expanded=False,
            )


st.sidebar.subheader("Past conversations")


if not threads:

    st.sidebar.write(
        "No past conversations yet."
    )

else:

    for index, thread_id in enumerate(threads, start=1):

        if st.sidebar.button(
            f"Chat {index}",
            key=f"side-thread-{thread_id}",
            use_container_width=True,
        ):
            selected_thread = thread_id


st.title("Chatbot")

st.caption(
    "Chat with your PDF, search the web, "
    "check stock prices, and perform calculations."
)


for message in st.session_state["message_history"]:

    with st.chat_message(message["role"]):
        st.write(message["content"])


user_input = st.chat_input(
    "Ask something..."
)


if user_input:

    st.session_state["message_history"].append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    with st.chat_message("user"):
        st.write(user_input)


    config = {
        "configurable": {
            "thread_id": thread_key,
        },
        "metadata": {
            "thread_id": thread_key,
        },
        "run_name": "chat_turn",
    }


    with st.chat_message("assistant"):

        status_holder = {
            "box": None
        }


        def ai_only_stream():

            for message_chunk, _ in chatbot.stream(
                {
                    "messages": [
                        HumanMessage(
                            content=user_input
                        )
                    ]
                },
                config=config,
                stream_mode="messages",
            ):

                if isinstance(
                    message_chunk,
                    ToolMessage,
                ):

                    tool_name = getattr(
                        message_chunk,
                        "name",
                        "tool",
                    )

                    if status_holder["box"] is None:

                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}`...",
                            expanded=True,
                        )

                    else:

                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}`...",
                            state="running",
                            expanded=True,
                        )


                if isinstance(
                    message_chunk,
                    AIMessage,
                ):

                    if isinstance(
                        message_chunk.content,
                        str,
                    ):
                        yield message_chunk.content


        ai_message = st.write_stream(
            ai_only_stream()
        )


        if status_holder["box"] is not None:

            status_holder["box"].update(
                label="✅ Tool finished",
                state="complete",
                expanded=False,
            )


    st.session_state["message_history"].append(
        {
            "role": "assistant",
            "content": ai_message,
        }
    )


    doc_meta = thread_document_metadata(
        thread_key
    )


    if doc_meta:

        st.caption(
            f"📄 {doc_meta.get('filename')} · "
            f"{doc_meta.get('chunks')} chunks · "
            f"{doc_meta.get('documents')} pages"
        )


if selected_thread:

    st.session_state["thread_id"] = selected_thread

    messages = load_conversation(
        selected_thread
    )

    temp_messages = []

    for message in messages:

        if isinstance(
            message,
            HumanMessage,
        ):
            role = "user"

        elif isinstance(
            message,
            AIMessage,
        ):
            role = "assistant"

        else:
            continue


        content = message.content

        if isinstance(content, str):

            temp_messages.append(
                {
                    "role": role,
                    "content": content,
                }
            )


    st.session_state["message_history"] = (
        temp_messages
    )

    st.session_state[
        "ingested_docs"
    ].setdefault(
        str(selected_thread),
        {},
    )

    st.rerun()