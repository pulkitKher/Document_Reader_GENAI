import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "QA-pdf-project")

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="PDF Question Answering", layout="centered")
st.title("PDF Question Answering App (Long Answers)")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    # Save uploaded PDF
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # ---------------- LOAD PDF ----------------
    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    # Combine all PDF text
    full_text = "\n".join(doc.page_content for doc in docs)

    st.success("PDF loaded successfully!")

    # ---------------- QUESTION INPUT ----------------
    question = st.text_input("Ask a question from the PDF")

    if question:
        with st.spinner("Generating detailed answer..."):

            # ---------------- PROMPT (LONG ANSWER STYLE) ----------------
            prompt = PromptTemplate.from_template(
                """
                You are answering a question from an academic textbook.

                Instructions:
                - Give a detailed and well-structured answer.
                - Explain the concept clearly and in depth.
                - Use proper scientific terminology.
                - Write in paragraph form.
                - The answer should be suitable for exams and notes.
                - Do NOT repeat the question.
                - Do NOT add meta phrases like "Here is the answer".

                Content:
                {content}

                Question:
                {question}
                """
            )

            # ---------------- LLM ----------------
            llm = Ollama(
                model="gemma:2b",
                temperature=0.2,
                num_predict=500 
            )

            parser = StrOutputParser()
            chain = prompt | llm | parser

            # Limit context for speed & stability
            response = chain.invoke({
                "content": full_text[:10000],
                "question": question
            })

            # ---------------- OUTPUT ----------------
            st.subheader("Answer")
            st.write(response)

else:
    st.info("Please upload a PDF to begin.")

