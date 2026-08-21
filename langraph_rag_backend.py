from __future__ import annotations

import os
import sqlite3
import tempfile
from typing import Annotated, Any, Optional, TypedDict

import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


_THREAD_RETRIEVERS: dict[str, Any] = {}
_THREAD_METADATA: dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    if not thread_id:
        return None

    return _THREAD_RETRIEVERS.get(str(thread_id))


def ingest_pdf(
    file_bytes: bytes,
    thread_id: str,
    filename: Optional[str] = None,
) -> dict:

    if not file_bytes:
        raise ValueError("No PDF data received.")

    thread_id = str(thread_id)

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf",
    ) as temp_file:

        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=[
                "\n\n",
                "\n",
                " ",
                "",
            ],
        )

        chunks = splitter.split_documents(
            documents
        )

        vector_store = FAISS.from_documents(
            chunks,
            embeddings,
        )

        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 4,
            },
        )

        _THREAD_RETRIEVERS[thread_id] = retriever

        metadata = {
            "filename": filename or "uploaded.pdf",
            "documents": len(documents),
            "chunks": len(chunks),
        }

        _THREAD_METADATA[thread_id] = metadata

        return metadata

    finally:

        try:
            os.remove(temp_path)
        except OSError:
            pass


search_tool = DuckDuckGoSearchRun(
    region="us-en"
)


@tool
def calculator(
    first_num: float,
    second_num: float,
    operation: str,
) -> dict:
    """
    Perform basic arithmetic operations.

    Supported operations:
    add, sub, mul, div
    """

    try:

        if operation == "add":
            result = first_num + second_num

        elif operation == "sub":
            result = first_num - second_num

        elif operation == "mul":
            result = first_num * second_num

        elif operation == "div":

            if second_num == 0:
                return {
                    "error": "Division by zero is not allowed."
                }

            result = first_num / second_num

        else:
            return {
                "error": (
                    f"Unsupported operation: {operation}"
                )
            }

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }

    except Exception as error:

        return {
            "error": str(error)
        }


@tool
def get_stock_price(
    symbol: str,
) -> dict:
    """
    Fetch the latest stock price using Alpha Vantage.
    """

    api_key = os.getenv(
        "ALPHA_VANTAGE_API_KEY"
    )

    if not api_key:
        return {
            "error": (
                "Alpha Vantage API key is not configured."
            )
        }

    url = (
        "https://www.alphavantage.co/query"
        "?function=GLOBAL_QUOTE"
        f"&symbol={symbol}"
        f"&apikey={api_key}"
    )

    response = requests.get(
        url,
        timeout=10,
    )

    response.raise_for_status()

    return response.json()


@tool
def rag_tool(
    query: str,
    thread_id: Optional[str] = None,
) -> dict:
    """
    Retrieve relevant information from the PDF
    uploaded in the current chat.
    """

    if not thread_id:
        return {
            "error": "Chat session information is missing.",
            "query": query,
        }

    retriever = _get_retriever(
        thread_id
    )

    if retriever is None:
        return {
            "error": (
                "No PDF is indexed for this chat. "
                "Please upload a PDF first."
            ),
            "query": query,
        }

    documents = retriever.invoke(
        query
    )

    context = [
        document.page_content
        for document in documents
    ]

    metadata = [
        document.metadata
        for document in documents
    ]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(
            str(thread_id),
            {},
        ).get("filename"),
    }


tools = [
    search_tool,
    get_stock_price,
    calculator,
    rag_tool,
]


llm_with_tools = llm.bind_tools(
    tools
)


class ChatState(TypedDict):
    messages: Annotated[
        list[BaseMessage],
        add_messages,
    ]


def chat_node(
    state: ChatState,
    config=None,
):

    thread_id = None

    if config and isinstance(config, dict):

        thread_id = config.get(
            "configurable",
            {},
        ).get(
            "thread_id"
        )

    system_message = SystemMessage(
        content=(
            "You are Chatbot, a helpful multi-utility "
            "AI assistant. "
            "For questions about an uploaded PDF, use "
            "the `rag_tool` and pass the current chat "
            "session identifier. "
            "You can use web search, stock price, and "
            "calculator tools when appropriate. "
            "If the user asks about a PDF and no PDF "
            "is indexed, ask them to upload a PDF."
        )
    )

    messages = [
        system_message,
        *state["messages"],
    ]

    response = llm_with_tools.invoke(
        messages,
        config=config,
    )

    return {
        "messages": [
            response
        ]
    }


tool_node = ToolNode(
    tools
)


connection = sqlite3.connect(
    "chatbot.db",
    check_same_thread=False,
)


checkpointer = SqliteSaver(
    conn=connection
)


graph = StateGraph(
    ChatState
)


graph.add_node(
    "chat_node",
    chat_node,
)

graph.add_node(
    "tools",
    tool_node,
)


graph.add_edge(
    START,
    "chat_node",
)


graph.add_conditional_edges(
    "chat_node",
    tools_condition,
)


graph.add_edge(
    "tools",
    "chat_node",
)


chatbot = graph.compile(
    checkpointer=checkpointer
)


def retrieve_all_threads():
    threads = set()

    for checkpoint in checkpointer.list(
        None
    ):

        thread_id = checkpoint.config[
            "configurable"
        ].get(
            "thread_id"
        )

        if thread_id:
            threads.add(
                thread_id
            )

    return list(threads)


def thread_has_document(
    thread_id: str,
) -> bool:

    return (
        str(thread_id)
        in _THREAD_RETRIEVERS
    )


def thread_document_metadata(
    thread_id: str,
) -> dict:

    return _THREAD_METADATA.get(
        str(thread_id),
        {},
    )