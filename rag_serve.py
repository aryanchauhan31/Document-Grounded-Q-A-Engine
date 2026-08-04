import ast
import operator
import os

from datasets import load_dataset
from pinecone import Pinecone
from ray import serve
from starlette.requests import Request
from starlette.responses import JSONResponse

from llama_index.core import Document, PromptTemplate, Settings, VectorStoreIndex, get_response_synthesizer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.workflow import Workflow, step, Context, Event, StartEvent, StopEvent
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.evaluation import FaithfulnessEvaluator
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.vector_stores.pinecone import PineconeVectorStore

MAX_RETRIES = 2
PINECONE_INDEX_NAME = "cuad-legal-rag"

QA_TEMPLATE = PromptTemplate(
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, answer the "
    "query in a complete, well-formed sentence that directly restates and "
    "addresses the question. If the context does not contain the answer, "
    "say so in a complete sentence.\n"
    "Query: {query_str}\n"
    "Answer: "
)
# Fixed pool of GPU-bound vLLM backends (one per GPU). Each stateless Serve
# replica round-robins its requests across this pool; replica count scales
# independently of backend count.
VLLM_BACKENDS = [
    "http://127.0.0.1:18000/v1",
    "http://127.0.0.1:18002/v1",
]

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", device="cpu")
Settings.embed_model = embed_model
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)

_CALC_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _CALC_OPS:
        return _CALC_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _CALC_OPS:
        return _CALC_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def calculate(expression: str) -> str:
    """Evaluate a basic arithmetic expression (+, -, *, /, **) and return the numeric result."""
    tree = ast.parse(expression, mode="eval")
    return str(_safe_eval(tree.body))


class RetrieveEvent(Event):
    query: str
    attempt: int


class GenerateEvent(Event):
    query: str
    nodes: list
    attempt: int


def build_index():
    # Dense leg: query the already-ingested Pinecone index directly (no
    # re-embedding/re-upsert here -- see ingest_cuad.py for the batch job
    # that populated it from the real CUAD contract corpus).
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    pinecone_vector_index = VectorStoreIndex.from_vector_store(vector_store)
    vector_retriever = pinecone_vector_index.as_retriever(similarity_top_k=5)

    # Sparse leg: BM25 needs raw node text locally, so rebuild the identical
    # chunking over the same real contracts (cheap -- no embedding involved).
    ds = load_dataset("chenghao/cuad_qa", split="train")
    contracts = {}
    for row in ds:
        contracts.setdefault(row["title"], row["context"])
    documents = [
        Document(text=text, metadata={"title": title})
        for title, text in contracts.items()
    ]
    nodes = Settings.node_parser.get_nodes_from_documents(documents)
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=5)

    return vector_retriever, bm25_retriever


def build_workflow(llm, vector_retriever, bm25_retriever):
    faith = FaithfulnessEvaluator(llm=llm)

    class RAGWorkflow(Workflow):
        @step
        async def retrieve(self, ctx: Context, ev: StartEvent | RetrieveEvent) -> GenerateEvent:
            query = ev.query
            attempt = getattr(ev, "attempt", 0)
            top_k = min(5 + attempt * 5, 10)
            fusion = QueryFusionRetriever(
                [vector_retriever, bm25_retriever],
                llm=llm,
                similarity_top_k=top_k,
                num_queries=1,
                mode="reciprocal_rerank",
            )
            nodes = await fusion.aretrieve(query)
            return GenerateEvent(query=query, nodes=nodes, attempt=attempt)

        @step
        async def generate(self, ctx: Context, ev: GenerateEvent) -> StopEvent | RetrieveEvent:
            synthesizer = get_response_synthesizer(
                llm=llm, response_mode="compact", text_qa_template=QA_TEMPLATE
            )
            response = synthesizer.synthesize(ev.query, nodes=ev.nodes)
            faith_result = await faith.aevaluate_response(query=ev.query, response=response)
            if faith_result.passing or ev.attempt >= MAX_RETRIES:
                return StopEvent(result=response)
            return RetrieveEvent(query=ev.query, attempt=ev.attempt + 1)

    return RAGWorkflow(timeout=120)


def build_backend(base_url, vector_retriever, bm25_retriever):
    llm = OpenAILike(
        model="HuggingFaceH4/zephyr-7b-alpha",
        api_base=base_url,
        api_key="dummy",
        max_tokens=256,
        is_chat_model=True,
        is_function_calling_model=False,
    )
    workflow = build_workflow(llm, vector_retriever, bm25_retriever)

    async def grounded_qa(query: str) -> str:
        """Answer a question using hybrid retrieval over the document corpus, with an automatic faithfulness-check retry loop."""
        response = await workflow.run(query=query)
        return str(response)

    qa_tool = FunctionTool.from_defaults(async_fn=grounded_qa, name="grounded_qa")
    calc_tool = FunctionTool.from_defaults(fn=calculate, name="calculator")

    agent = ReActAgent(
        tools=[qa_tool, calc_tool],
        llm=llm,
        system_prompt=(
            "You are a document-grounded QA assistant. Use grounded_qa for factual "
            "questions about the documents, and calculator for arithmetic."
        ),
    )
    return workflow, agent


@serve.deployment(
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 4,
        "target_ongoing_requests": 8,
    },
    ray_actor_options={"num_cpus": 1},
)
class RAGService:
    def __init__(self):
        # Built fresh per replica at actor startup (not at class-decoration
        # time) so Ray never needs to pickle live LLM/agent objects.
        vector_retriever, bm25_retriever = build_index()
        self.backends = [
            build_backend(url, vector_retriever, bm25_retriever) for url in VLLM_BACKENDS
        ]
        self._counter = 0

    def _pick_backend(self):
        backend = self.backends[self._counter % len(self.backends)]
        self._counter += 1
        return backend

    async def __call__(self, request: Request):
        path = request.url.path

        if path == "/health":
            return JSONResponse({"status": "ok"})

        body = await request.json()
        question = body.get("question", "")

        if path == "/rag":
            workflow, _ = self._pick_backend()
            response = await workflow.run(query=question)
            return JSONResponse({"answer": str(response)})

        if path == "/query":
            _, agent = self._pick_backend()
            result = await agent.run(user_msg=question)
            answer = getattr(result.response, "content", None) or str(result)
            return JSONResponse({"answer": answer})

        return JSONResponse({"error": "not found"}, status_code=404)


rag_app = RAGService.bind()
