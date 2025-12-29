
!pip install -Uq llama-index-core llama-index-llms-vllm llama-index-embeddings-huggingface vllm

import time
import json
import re
import os
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.vllm import Vllm
from llama_index.core.evaluation import FaithfulnessEvaluator, AnswerRelevancyEvaluator


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

qe = index.as_query_engine(response_mode="compact")

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

faith = FaithfulnessEvaluator(llm=llm)
rel = AnswerRelevancyEvaluator(llm=llm)

records = []
print("\n--- Starting Evaluation ---")
for q in questions[:5]: 
    resp = qe.query(q["text"])
    
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
