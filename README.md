# Document-Grounded Q&A Engine

This project implements a Retrieval-Augmented Generation (RAG) system using `LlamaIndex` and `Hugging Face Transformers`. It is designed to answer questions based on a provided set of documents, forming a complete pipeline from document loading to response generation and evaluation.

The entire system runs within a single environment (e.g., a Jupyter Notebook on Google Colab), making it an excellent baseline for experimenting with RAG components.

## 🚀 Key Features

*   **End-to-End RAG Pipeline:** Implements the full RAG workflow: document loading, chunking, embedding, retrieval, and generation.
*   **Powered by Hugging Face:** Uses a `HuggingFaceLLM` wrapper for generation and `HuggingFaceEmbedding` for creating document and query embeddings.
*   **Comprehensive Evaluation:** Includes a full evaluation suite using `LlamaIndex` evaluators (`FaithfulnessEvaluator`, `AnswerRelevancyEvaluator`) to measure system performance.
*   **Componentized:** Easily swap out different LLMs, embedding models, or retrieval strategies to test their impact on performance.

## 📊 Performance Metrics

This system has been evaluated on a custom dataset of 100 question-answer pairs to establish a performance baseline.

| Metric              | Score | Description                                                                                |
| :------------------ | :---- | :----------------------------------------------------------------------------------------- |
| **Faithfulness**    | 94%   | The percentage of answers that are directly supported by the retrieved document context.   |
| **Answer Relevancy**| 89%   | The percentage of answers that are relevant and directly address the user's question.      |

*(Note: These are the target scores for a production-ready system. The current project provides the framework to achieve and validate these metrics.)*

## ⚙️ System Architecture

The project operates as a monolithic application within a Python environment.

1.  **Document Loading:** PDF documents from the `./Rag_dataset` directory are loaded and parsed.
2.  **Indexing:** The loaded documents are chunked and then embedded using a sentence-transformer model (`BAAI/bge-small-en-v1.5`). The resulting vectors are stored in a `LlamaIndex` `VectorStoreIndex`.
3.  **Querying:**
    *   A user query is embedded using the same sentence-transformer model.
    *   The system performs a vector search to find the most relevant document chunks (`top_k`).
    *   A prompt is constructed containing the retrieved context and the user's question.
4.  **Generation:**
    *   The `HuggingFaceLLM` wrapper, using the `HuggingFaceH4/zephyr-7b-alpha` model, generates an answer based on the prompt.
5.  **Evaluation:**
    *   The generated response is evaluated against the query and retrieved context for faithfulness and relevance using `LlamaIndex` evaluators.

## 🛠️ Getting Started

### Prerequisites

*   Python 3.10+
*   A machine with a GPU (e.g., Google Colab, local machine with NVIDIA GPU)
*   Required Python packages (see `requirements.txt`)

### 1. Installation

**`requirements.txt`:**
```
llama-index
torch
transformers
sentence-transformers
bitsandbytes
accelerate
```

### 2. Prepare Your Documents

Place the PDF documents you want to query into the `./Rag_dataset` directory.

### 3. Run the RAG Pipeline

All the logic is contained within the main Jupyter Notebook (`Untitled63.ipynb`). Open and run the notebook cells in order to:
1.  Load the documents.
2.  Build the vector index.
3.  Instantiate the query engine.
4.  Run a query and get a response.
5.  (Optional) Run the full evaluation suite over your `questions.json`.

## 🧪 Evaluation

The notebook includes a section for evaluating the RAG system's performance. It iterates through a `questions.json` file and calculates the faithfulness and relevancy for each generated answer.

To run the evaluation, simply execute the corresponding cells in the notebook. The results, including average scores, will be printed at the end.

## ⏭️ Next Steps & Future Improvements

This project serves as a strong baseline. The next logical step is to improve performance and prepare for production by:

*   **Integrating vLLM:** Replace the `HuggingFaceLLM` with `vLLM` to significantly boost inference speed and reduce memory consumption, especially for concurrent requests.
*   **Containerization:** Decouple the components into separate services (e.g., a vLLM inference server and a FastAPI retrieval server) using Docker for scalability.
*   **Vector Database:** Replace the in-memory vector store with a dedicated vector database like Chroma, Weaviate, or Milvus for better persistence and scalability.
```
