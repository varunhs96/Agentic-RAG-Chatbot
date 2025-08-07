

import os
import uuid
import asyncio
import streamlit as st
import threading
from ingestion_agent import IngestionAgent
from retrieval_agent import RetrievalAgent
from llm_response_agent import LLMResponseAgent
from mcp_bus import MCPBus
import time


answers = {}


mcp_bus = MCPBus()


ingestor = IngestionAgent()
retriever = RetrievalAgent()
llm_agent = LLMResponseAgent()


async def ingestion_handler(msg):
    trace_id = msg["trace_id"]
    file_objs = msg["payload"]["files"]
    file_paths = []


    for f in file_objs:
        path = f"temp_uploads/{trace_id}_{f.name}"
        os.makedirs("temp_uploads", exist_ok=True)
        with open(path, "wb") as out:
            out.write(f.read())
        file_paths.append(path)

    chunked_docs = ingestor.ingest(file_paths)

    await mcp_bus.send({
        "sender": "IngestionAgent",
        "receiver": "RetrievalAgent",
        "type": "DOCUMENTS_PARSED",
        "trace_id": trace_id,
        "payload": {"chunked_docs": chunked_docs}
    })

async def retrieval_handler(msg):
    trace_id = msg["trace_id"]

    if msg["type"] == "DOCUMENTS_PARSED":
        retriever.build_index(msg["payload"]["chunked_docs"])

    elif msg["type"] == "QUERY":
        top_chunks = retriever.query(msg["payload"]["query"])
        await mcp_bus.send({
            "sender": "RetrievalAgent",
            "receiver": "LLMResponseAgent",
            "type": "CONTEXT_RESPONSE",
            "trace_id": trace_id,
            "payload": {
                "query": msg["payload"]["query"],
                "top_chunks": top_chunks
            }
        })


async def llm_handler(msg):
    trace_id = msg["trace_id"]
    response = llm_agent.generate_answer(
        msg["payload"]["query"],
        msg["payload"]["top_chunks"]
    )
    answers[trace_id] = response


mcp_bus.subscribe("IngestionAgent", ingestion_handler)
mcp_bus.subscribe("RetrievalAgent", retrieval_handler)
mcp_bus.subscribe("LLMResponseAgent", llm_handler)


def start_event_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(mcp_bus.run())
    loop.run_forever()

threading.Thread(target=start_event_loop, daemon=True).start()


st.title("📚 Agentic RAG Chatbot")
st.markdown("Upload your documents, ask questions, and get context-aware answers.")

uploaded_files = st.file_uploader("Upload documents", type=["pdf", "docx", "pptx", "csv", "txt", "md"], accept_multiple_files=True)

if uploaded_files:
    if st.button("Ingest Documents"):
        trace_id = str(uuid.uuid4())
        asyncio.run(mcp_bus.send({
            "sender": "UI",
            "receiver": "IngestionAgent",
            "type": "UPLOAD",
            "trace_id": trace_id,
            "payload": {"files": uploaded_files}
        }))
        st.success("Documents sent for ingestion!")

query = st.text_input("Ask a question about your documents")

if query:
    if st.button("Get Answer"):
        trace_id = str(uuid.uuid4())
        asyncio.run(mcp_bus.send({
            "sender": "UI",
            "receiver": "RetrievalAgent",
            "type": "QUERY",
            "trace_id": trace_id,
            "payload": {"query": query}
        }))

        with st.spinner("Thinking..."):
            for _ in range(30):
                asyncio.sleep(0.3)
                if trace_id in answers:
                    st.markdown("### 💬 Answer:")
                    st.write(answers[trace_id])
                    break
            else:
                st.error("No response received. Try again.")