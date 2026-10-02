from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path
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
from langgraph.prebuilt import InjectedState, ToolNode, tools_condition


load_dotenv()


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


BASE_DIR = Path(__file__).resolve().parent

DATABASE_PATH = BASE_DIR / "chatbot.db"

FAISS_BASE_DIR = BASE_DIR / "faiss_indexes"

FAISS_BASE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


_THREAD_RETRIEVERS: dict[str, Any] = {}


connection = sqlite3.connect(
    str(DATABASE_PATH),
    check_same_thread=False,
)


connection.execute(
    """
    CREATE TABLE IF NOT EXISTS thread_documents (
        thread_id TEXT PRIMARY KEY,
        filename TEXT NOT NULL,
        documents INTEGER NOT NULL,
        chunks INTEGER NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """
)

connection.commit()


def _get_faiss_dir(thread_id: str):

    return FAISS_BASE_DIR / str(thread_id)


def _get_retriever(thread_id: Optional[str]):

    if not thread_id:
        return None

    thread_id = str(thread_id)

    retriever = _THREAD_RETRIEVERS.get(
        thread_id
    )

    if retriever is not None:
        return retriever

    index_dir = _get_faiss_dir(
        thread_id
    )

    if not (index_dir / "index.faiss").exists():
        return None

    if not (index_dir / "index.pkl").exists():
        return None

    vector_store = FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 4,
        },
    )

    _THREAD_RETRIEVERS[
        thread_id
    ] = retriever

    return retriever


def ingest_pdf(
    file_bytes: bytes,
    thread_id: str,
    filename: Optional[str] = None,
) -> dict:

    if not file_bytes:
        raise ValueError("No PDF data received.")

    thread_id = str(thread_id)

    filename = filename or "uploaded.pdf"

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf",
    ) as temp_file:

        temp_file.write(file_bytes)

        temp_path = temp_file.name

    try:

        loader = PyPDFLoader(
            temp_path
        )

        documents = loader.load()

        if not documents:
            raise ValueError(
                "Could not extract any content from the PDF."
            )

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

        if not chunks:
            raise ValueError(
                "PDF could not be split into chunks."
            )

        vector_store = FAISS.from_documents(
            chunks,
            embeddings,
        )

        index_dir = _get_faiss_dir(
            thread_id
        )

        index_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        vector_store.save_local(
            str(index_dir)
        )

        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 4,
            },
        )

        _THREAD_RETRIEVERS[
            thread_id
        ] = retriever

        connection.execute(
            """
            INSERT OR REPLACE INTO thread_documents
            (
                thread_id,
                filename,
                documents,
                chunks
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                thread_id,
                filename,
                len(documents),
                len(chunks),
            ),
        )

        connection.commit()

        return {
            "filename": filename,
            "documents": len(documents),
            "chunks": len(chunks),
        }

    finally:

        try:
            os.remove(temp_path)
        except OSError:
            pass


def thread_document_metadata(
    thread_id: str,
) -> dict:

    cursor = connection.execute(
        """
        SELECT
            filename,
            documents,
            chunks
        FROM thread_documents
        WHERE thread_id = ?
        """,
        (
            str(thread_id),
        ),
    )

    row = cursor.fetchone()

    if not row:
        return {}

    return {
        "filename": row[0],
        "documents": row[1],
        "chunks": row[2],
    }


def thread_has_document(
    thread_id: str,
) -> bool:

    return bool(
        thread_document_metadata(
            thread_id
        )
    )


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

    try:

        response = requests.get(
            url,
            timeout=10,
        )

        response.raise_for_status()

        return response.json()

    except Exception as error:

        return {
            "error": str(error)
        }


@tool
def rag_tool(
    query: str,
    state: Annotated[
        dict,
        InjectedState,
    ],
) -> dict:
    """
    Retrieve relevant information from the PDF
    uploaded in the current chat.
    """

    thread_id = state.get(
        "thread_id"
    )

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

    try:

        documents = retriever.invoke(
            query
        )

    except Exception as error:

        return {
            "error": f"PDF search failed: {error}",
            "query": query,
        }

    context = [
        document.page_content
        for document in documents
    ]

    metadata = [
        document.metadata
        for document in documents
    ]

    document_metadata = thread_document_metadata(
        thread_id
    )

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": document_metadata.get(
            "filename"
        ),
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

    thread_id: str


def chat_node(
    state: ChatState,
    config=None,
):

    system_message = SystemMessage(
        content=(
            "You are Chatbot, a helpful multi-utility "
            "AI assistant. "
            "For questions about an uploaded PDF, use "
            "the `rag_tool`. "
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

    try:

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

    except Exception:

        pass

    return list(threads)
