!pip install -Uq llama-index-embeddings-huggingface

import huggingface_hub
huggingface_hub.login()

from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import Settings

llm = HuggingFaceLLM(model_name = 'HuggingFaceH4/zephyr-7b-alpha',
                     tokenizer_name = 'HuggingFaceH4/zephyr-7b-alpha',
                     query_wrapper_prompt = '{query_str}',
                     context_window = 3900,
                     max_new_tokens = 256,
                    #  model_kwargs = {'quantization_config': {'load_in_4bit': True}},
                     generate_kwargs = {'temperature': 0.7, 'top_k': 50, 'top_p':0.95},
                     device_map= 'auto',
                     )

!mkdir Rag_dataset
!unzip  Rag_dataset.zip -d /content/Rag_dataset

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = embed_model  # set global default so VectorStoreIndex uses it

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

docs = SimpleDirectoryReader(
    "./Rag_dataset",
    recursive=True  
).load_data()  
index = VectorStoreIndex.from_documents(docs)

!mkdir storage
index.storage_context.persist(persist_dir="./storage")

Settings.llm = llm

mport json, re, os
from llama_index.core import StorageContext, load_index_from_storage

storage = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage)

qe = index.as_query_engine(response_mode="compact")

with open("/content/Rag_dataset/questions.json", "r") as f:
    questions = json.load(f)

def normalize_answer(kind, text):
    s = str(text).strip()
    if kind == "boolean":
        return "True" if re.search(r"\\b(yes|true)\\b", s, re.I) else "False"
    if kind == "number":
        m = re.search(r"[-+]?[0-9][0-9,]*(?:\\.[0-9]+)?", s)
        return m.group(0).replace(",", "") if m else "N/A"
    if kind in ("name", "names"):
        return s if s else "N/A"
    return s

outputs = []
for i, q in enumerate(questions, 1):
    prompt = q.get("text", "")
    kind = q.get("kind", "")
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

from llama_index.core.evaluation import FaithfulnessEvaluator, AnswerRelevancyEvaluator

faith = FaithfulnessEvaluator(llm=llm)
rel = AnswerRelevancyEvaluator(llm=llm)

records = []
for q in questions:
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
