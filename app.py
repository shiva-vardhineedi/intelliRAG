import streamlit as st
import os
import PyPDF2
import faiss
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import time
import redis
from groq import Groq
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import logging
from pprint import pprint
import pika

# Load environment variables from .env file
load_dotenv()

# FAISS Index setup
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(EMBEDDING_MODEL)

INDEX_DIR = "faiss_indexes"

# Initialize the ChatGroq models using API key from environment variable
summary_llm = ChatGroq(api_key=os.getenv('GROQ_API_KEY'), max_tokens=100)  # Set max tokens for summary response length
llm = ChatGroq(api_key=os.getenv('GROQ_API_KEY'), max_tokens=400)  # Set max tokens for the final response

# Configure Redis client
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Configure logging with pretty print
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()
pp = pprint

# Configure RabbitMQ connection
rabbitmq_host = 'localhost'
connection = pika.BlockingConnection(pika.ConnectionParameters(host=rabbitmq_host))
channel = connection.channel()
channel.queue_declare(queue='question_queue')

# Function to create index
def create_index(directory, pdf_files):
    # Check if multiple files are selected
    if len(pdf_files) > 1:
        index_filename = os.path.join(directory, "combined_index.faiss")
        chunks_filename = os.path.join(directory, "combined_chunks.pkl")
        if os.path.exists(index_filename) and os.path.exists(chunks_filename):
            st.info("Combined index already exists. Skipping creation.")
            st.session_state.index_created = True
            return

        text_chunks = []
        embeddings = []

        progress_bar = st.progress(0)
        for i, pdf_file in enumerate(pdf_files):
            file_path = os.path.join(directory, pdf_file.name)
            with open(file_path, "wb") as f:
                f.write(pdf_file.read())

            try:
                pdf_reader = PyPDF2.PdfReader(file_path)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    if text:
                        text_chunks.append(text)
                        embeddings.append(model.encode(text))
            except Exception as e:
                st.error(f"Error reading PDF file {pdf_file.name}: {e}")
                return

            progress_bar.progress((i + 1) / len(pdf_files))

        if len(embeddings) == 0:
            st.error("No valid text found in the provided PDFs. Please upload PDFs with extractable text.")
            return

        dimension = embeddings[0].shape[0]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings, dtype="float32"))

        # Save index to disk
        try:
            faiss.write_index(index, index_filename)
            with open(chunks_filename, "wb") as f:
                pickle.dump(text_chunks, f)
        except Exception as e:
            st.error(f"Error saving index or chunks: {e}")
            return

        progress_bar.empty()
        st.success("Combined index created and saved successfully!")
        st.session_state.index_created = True

    # If only one file is selected
    else:
        pdf_file = pdf_files[0]
        index_filename = os.path.join(directory, f"{pdf_file.name}_index.faiss")
        chunks_filename = os.path.join(directory, f"{pdf_file.name}_chunks.pkl")
        if os.path.exists(index_filename) and os.path.exists(chunks_filename):
            st.info(f"Index for {pdf_file.name} already exists. Skipping creation.")
            st.session_state.index_created = True
            return

        text_chunks = []
        embeddings = []

        progress_bar = st.progress(0)
        file_path = os.path.join(directory, pdf_file.name)
        with open(file_path, "wb") as f:
            f.write(pdf_file.read())

        try:
            pdf_reader = PyPDF2.PdfReader(file_path)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                if text:
                    text_chunks.append(text)
                    embeddings.append(model.encode(text))
        except Exception as e:
            st.error(f"Error reading PDF file {pdf_file.name}: {e}")
            return

        progress_bar.progress(1.0)

        if len(embeddings) == 0:
            st.error("No valid text found in the provided PDF. Please upload a PDF with extractable text.")
            return

        dimension = embeddings[0].shape[0]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings, dtype="float32"))

        # Save index to disk
        try:
            faiss.write_index(index, index_filename)
            with open(chunks_filename, "wb") as f:
                pickle.dump(text_chunks, f)
        except Exception as e:
            st.error(f"Error saving index or chunks: {e}")
            return

        progress_bar.empty()
        st.success(f"Index for {pdf_file.name} created and saved successfully!")
        st.session_state.index_created = True

# Function to summarize each embedding using ChatGroq
def summarize_chunk(chunk):
    prompt = (
        f"Here is a document chunk:\n\n{chunk}\n\n"
        "Please summarize this chunk by retaining all critical information but keeping the summary concise. "
        "The summary must be brief and not exceed 50 tokens. Avoid any unnecessary details."
    )

    response = summary_llm.invoke([("user", prompt)])
    return response.content.strip()

# Function to answer the question using summarized embeddings
def query_system(query, index_path, chunks_path):
    # Load index and chunks
    try:
        index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            text_chunks = pickle.load(f)
    except Exception as e:
        return f"Error loading index or chunks: {e}"

    # Check if the query result is already in Redis cache
    cached_response = redis_client.get(query)  # Fetch response from Redis cache
    if cached_response:
        logger.info("Cache hit for query: Returning cached response.")
        return cached_response.decode('utf-8')
    else:
        logger.info("Cache miss for query: Proceeding with processing.")
    
    logger.info(f"Received query: {query}")

    # Push the query to RabbitMQ queue
    channel.basic_publish(exchange='', routing_key='question_queue', body=query)
    logger.info(f"Query pushed to RabbitMQ queue: {query}")

    # Encode the question
    question_embedding = model.encode([query])[0]
    
    # Retrieve relevant context using the FAISS index
    D, I = index.search(np.array([question_embedding], dtype="float32"), k=5)
    relevant_texts = [text_chunks[idx] for idx in I[0] if idx != -1]

    # Summarize each chunk individually
    summarized_chunks = []
    for i, chunk in enumerate(relevant_texts):
        with st.spinner(f"Summarizing chunk {i+1} of {len(relevant_texts)}..."):
            summary = summarize_chunk(chunk)
            summarized_chunks.append(summary)

    # Combine all summarized chunks
    combined_summary = "\n\n".join(summarized_chunks)

    # Prepare the messages for ChatGroq invocation with updated prompt for styled and concise responses
    messages = [
        (
            "system",
            "You are a helpful assistant that provides answers in a well-structured, styled format. "
            "Use bullet points, lists, or sections to make the response easy to read. Keep your response concise, "
            "only provide the most critical information needed to answer the user's question. "
            "Avoid excessive details, and focus on clarity and brevity.\n\n"
            "Context:\n"
            f"{combined_summary}\n\n"
            "Note: The following information was retrieved internally from the FAISS index."
        ),
        ("user", query)
    ]

    # Invoke the model with the provided messages
    logger.info("Invoking the ChatGroq model with the provided context and query...")
    response = llm.invoke(messages)

    # Cache the response in Redis
    redis_client.set(query, response.content)

    pp(response.content)  # Pretty print the final response for better debugging

    return response.content


st.title("RAG Implementation with Streamlit")

st.sidebar.title("Sections")
selection = st.sidebar.radio("Select Section", ["Index Creation", "Question Answering", "Display Cache"])

if selection == "Index Creation":
    st.header("Create Index for your PDF Documents")
    if not os.path.exists(INDEX_DIR):
        os.makedirs(INDEX_DIR)

    uploaded_pdfs = st.file_uploader("Upload your PDF Documents", type=["pdf"], accept_multiple_files=True)

    if uploaded_pdfs:
        if st.button("Create Index"):
            create_index(INDEX_DIR, uploaded_pdfs)
            st.session_state.index_created = True

    if "index_created" in st.session_state and st.session_state.index_created:
        if st.button("Proceed to Question Answering Stage"):
            selection = "Question Answering"

elif selection == "Question Answering":
    # Handle Question Answering section
    st.header("Ask Questions based on your Indexed Data")
    index_files = [f for f in os.listdir(INDEX_DIR) if f.endswith("_index.faiss")]  # Get the list of available FAISS indexes

    if len(index_files) > 0:
        selected_index_file = st.selectbox("Select FAISS Index", index_files)

        if selected_index_file:
            index_path = os.path.join(INDEX_DIR, selected_index_file)
            try:
                index = faiss.read_index(index_path)
            except Exception as e:
                st.error(f"Error loading FAISS index: {e}")
                index = None

            chunks_filename = selected_index_file.replace("_index.faiss", "_chunks.pkl")
            chunks_path = os.path.join(INDEX_DIR, chunks_filename)

            if os.path.exists(chunks_path):
                with open(chunks_path, "rb") as f:
                    text_chunks = pickle.load(f)
            else:
                st.error("Chunks file not found. Please ensure the index was created correctly.")
                text_chunks = None

            user_question = st.text_input("Enter your question:")

            if st.button("Search Answer") and user_question:
                if os.path.exists(index_path) and os.path.exists(chunks_path):
                    if index is not None and text_chunks is not None:
                        with st.spinner("Searching for the best answers..."):
                            try:
                                answer = query_system(user_question, index_path, chunks_path)
                                st.write(f"Answer:\n\n{answer}")
                            except Exception as e:
                                st.error(f"Error during search: {e}")
                    else:
                        st.error("Cannot search as the index or chunks file was not loaded successfully.")
                else:
                    st.error("Cannot search as the index or chunks file was not loaded successfully.")
    else:
        st.warning("No indexes available. Please create an index first in the 'Index Creation' section.")

elif selection == "Display Cache":
    st.header("Display Top 5 Cached Q/A Pairs")
    cached_keys = redis_client.keys()
    if len(cached_keys) == 0:
        st.write("No cached Q/A pairs found.")
    else:
        top_keys = cached_keys[:5]  # Get the top 5 keys
        for key in top_keys:
            question = key.decode('utf-8')
            answer = redis_client.get(key).decode('utf-8')
            st.write(f"**Question:** {question}")
            st.write(f"**Answer:** {answer}")
            st.write("---")
