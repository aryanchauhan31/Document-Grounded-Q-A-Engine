
!pip install -Uq llama-index-core llama-index-llms-vllm llama-index-embeddings-huggingface llama-index-retrievers-bm25 arize-phoenix vllm

import ast
import asyncio
import operator
import time
import json
import re
import os
import phoenix as px
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage, get_response_synthesizer, set_global_handler
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.workflow import Workflow, step, Context, Event, StartEvent, StopEvent
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.vllm import Vllm
from llama_index.core.evaluation import FaithfulnessEvaluator, AnswerRelevancyEvaluator

px.launch_app()
set_global_handler("arize_phoenix")


llm = Vllm(
    model="HuggingFaceH4/zephyr-7b-alpha",
    trust_remote_code=True,
    tensor_parallel_size=1,  
    max_new_tokens=256,
    vllm_kwargs={
        "swap_space": 4,
        "gpu_memory_utilization": 0.90,
        "max_model_len": 4096,
        "dtype": "bfloat16",          
        "enable_prefix_caching": True 
    }
)

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = embed_model
Settings.llm = llm
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)


!mkdir -p Rag_dataset

if not os.path.exists("/content/Rag_dataset/questions.json"):
    !unzip -o Rag_dataset.zip -d /content/Rag_dataset

if not os.path.exists("./storage"):
    docs = SimpleDirectoryReader("./Rag_dataset", recursive=True).load_data()
    index = VectorStoreIndex.from_documents(docs)
    index.storage_context.persist(persist_dir="./storage")
else:
    storage = StorageContext.from_defaults(persist_dir="./storage")
    index = load_index_from_storage(storage)

vector_retriever = index.as_retriever(similarity_top_k=10)
bm25_retriever = BM25Retriever.from_defaults(docstore=index.docstore, similarity_top_k=10)

faith = FaithfulnessEvaluator(llm=llm)

MAX_RETRIES = 2


class RetrieveEvent(Event):
    query: str
    attempt: int


class GenerateEvent(Event):
    query: str
    nodes: list
    attempt: int


class RAGWorkflow(Workflow):
    @step
    async def retrieve(self, ctx: Context, ev: StartEvent | RetrieveEvent) -> GenerateEvent:
        query = ev.query
        attempt = getattr(ev, "attempt", 0)
        top_k = min(5 + attempt * 5, 10)
        fusion = QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            similarity_top_k=top_k,
            num_queries=1,
            mode="reciprocal_rerank",
        )
        nodes = await fusion.aretrieve(query)
        return GenerateEvent(query=query, nodes=nodes, attempt=attempt)

    @step
    async def generate(self, ctx: Context, ev: GenerateEvent) -> StopEvent | RetrieveEvent:
        synthesizer = get_response_synthesizer(response_mode="compact")
        response = synthesizer.synthesize(ev.query, nodes=ev.nodes)

        faith_result = await faith.aevaluate_response(query=ev.query, response=response)

        if faith_result.passing or ev.attempt >= MAX_RETRIES:
            return StopEvent(result=response)

        return RetrieveEvent(query=ev.query, attempt=ev.attempt + 1)


rag_workflow = RAGWorkflow(timeout=120)

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


async def grounded_qa(query: str) -> str:
    """Answer a question using hybrid (vector + BM25) retrieval over the document corpus, with an automatic faithfulness-check retry loop. Use this for any question requiring facts from the documents."""
    response = await rag_workflow.run(query=query)
    return str(response)


qa_tool = FunctionTool.from_defaults(async_fn=grounded_qa, name="grounded_qa")
calc_tool = FunctionTool.from_defaults(fn=calculate, name="calculator")

agent = ReActAgent(
    tools=[qa_tool, calc_tool],
    llm=llm,
    system_prompt=(
        "You are a document-grounded QA assistant. Always call the grounded_qa tool to "
        "retrieve and answer factual questions from the corpus before responding. Use the "
        "calculator tool only to combine or verify numbers already returned by grounded_qa. "
        "Never answer from your own knowledge without calling grounded_qa first."
    ),
)


class QueryEngineAdapter:
    """Lets the tool-using agent be called like a query engine's .query()."""

    def query(self, prompt):
        result = asyncio.run(agent.run(user_msg=prompt))
        return getattr(result.response, "content", None) or str(result)


qe = QueryEngineAdapter()

def benchmark_inference(query_engine, questions, limit=5):
    """
    Measures Latency and Throughput (Tokens/Sec) to validate performance gains.
    """
    print(f"\n--- Running Benchmark on first {limit} questions ---")
    total_time = 0
    total_tokens = 0

    for i, q in enumerate(questions[:limit]):
        prompt = q.get("text", "")
        
        start_t = time.perf_counter()
        response = query_engine.query(prompt)
        end_t = time.perf_counter()
        
        duration = end_t - start_t
        output_tokens = len(str(response)) / 4 
        
        total_time += duration
        total_tokens += output_tokens
        
        print(f"Q{i+1}: {duration:.4f}s | Est. Tokens: {int(output_tokens)}")

    avg_latency = total_time / limit
    throughput = total_tokens / total_time
    
    print(f"\nResults:")
    print(f"Average Latency: {avg_latency:.4f} s/query")
    print(f"Est. Throughput: {throughput:.2f} tokens/s")
    print("---------------------------------------------------\n")


with open("/content/Rag_dataset/questions.json", "r") as f:
    questions = json.load(f)


benchmark_inference(qe, questions)

def normalize_answer(kind, text):
    s = str(text).strip()
    if kind == "boolean":
        return "True" if re.search(r"\b(yes|true)\b", s, re.I) else "False"
    if kind == "number":
        m = re.search(r"[-+]?[0-9][0-9,]*(?:\.[0-9]+)?", s)
        return m.group(0).replace(",", "") if m else "N/A"
    if kind in ("name", "names"):
        return s if s else "N/A"
    return s

outputs = []
for i, q in enumerate(questions, 1):
    prompt = q.get("text", "")
    kind = q.get("kind", "")
    
    # Querying
    resp = qe.query(prompt)
    raw = str(resp)
    
    outputs.append({
        "idx": i,
        "question": prompt,
        "kind": kind,
        "raw": raw,
        "answer": normalize_answer(kind, raw)
    })

with open("answers.json", "w") as f:
    json.dump(outputs, f, indent=2)

print(f"Wrote answers.json with {len(outputs)} rows")

rel = AnswerRelevancyEvaluator(llm=llm)

records = []
print("\n--- Starting Evaluation ---")
for q in questions[:5]:
    resp = asyncio.run(rag_workflow.run(query=q["text"]))

    f = faith.evaluate_response(query=q["text"], response=resp)
    r = rel.evaluate_response(query=q["text"], response=resp)
    
    records.append({
        "q": q["text"],
        "faithful": f.passing,
        "faith_score": f.score,
        "relevant": r.passing,
        "rel_score": r.score
    })
    print(f"Evaluated: {q['text'][:30]}... | Faith: {f.score} | Rel: {r.score}")


if records:
    avg_faith = sum(r['faith_score'] for r in records) / len(records)
    avg_rel = sum(r['rel_score'] for r in records) / len(records)
    print(f"\nFinal Metrics -> Faithfulness: {avg_faith:.2f}, Relevancy: {avg_rel:.2f}")
