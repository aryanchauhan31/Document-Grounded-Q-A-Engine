import asyncio
import json
import os

from pinecone import Pinecone

from llama_index.core import PromptTemplate, Settings, VectorStoreIndex, get_response_synthesizer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.evaluation import FaithfulnessEvaluator, AnswerRelevancyEvaluator
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.vector_stores.pinecone import PineconeVectorStore

N_EVAL = 15

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

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", device="cpu")
Settings.embed_model = embed_model
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)

llm = OpenAILike(
    model="HuggingFaceH4/zephyr-7b-alpha",
    api_base="http://127.0.0.1:18000/v1",
    api_key="dummy",
    max_tokens=256,
    is_chat_model=True,
    is_function_calling_model=False,
)

judge_llm = OpenAILike(
    model="Qwen/Qwen2.5-14B-Instruct",
    api_base="http://127.0.0.1:18003/v1",
    api_key="dummy",
    max_tokens=256,
    is_chat_model=True,
    is_function_calling_model=False,
)

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
pinecone_index = pc.Index("cuad-legal-rag")
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
pinecone_vector_index = VectorStoreIndex.from_vector_store(vector_store)
vector_retriever = pinecone_vector_index.as_retriever(similarity_top_k=5)

with open("/workspace/cuad_eval_questions.json") as f:
    eval_rows = json.load(f)[:N_EVAL]

faith = FaithfulnessEvaluator(llm=judge_llm)
rel = AnswerRelevancyEvaluator(llm=judge_llm)


async def main():
    synthesizer = get_response_synthesizer(
        llm=llm, response_mode="compact", text_qa_template=QA_TEMPLATE
    )
    records = []
    for i, row in enumerate(eval_rows, 1):
        nodes = await vector_retriever.aretrieve(row["question"])
        response = synthesizer.synthesize(row["question"], nodes=nodes)

        f_result = await faith.aevaluate_response(query=row["question"], response=response)
        r_result = await rel.aevaluate_response(query=row["question"], response=response)

        records.append({
            "question": row["question"],
            "expected": row["expected_answer"],
            "got": str(response),
            "faithful": f_result.passing,
            "faith_score": f_result.score,
            "relevant": r_result.passing,
            "rel_score": r_result.score,
        })
        print(f"[{i}/{len(eval_rows)}] faith={f_result.score} rel={r_result.score} | {row['question'][:70]}")

    faith_scores = [r["faith_score"] for r in records if r["faith_score"] is not None]
    rel_scores = [r["rel_score"] for r in records if r["rel_score"] is not None]
    avg_faith = sum(faith_scores) / len(faith_scores) if faith_scores else float("nan")
    avg_rel = sum(rel_scores) / len(rel_scores) if rel_scores else float("nan")
    faith_pass_rate = sum(1 for r in records if r["faithful"]) / len(records)
    rel_pass_rate = sum(1 for r in records if r["relevant"]) / len(records)

    print(f"\n=== CUAD Eval Results (n={len(records)}) ===")
    print(f"Avg Faithfulness: {avg_faith:.2f} | Pass rate: {faith_pass_rate:.0%} | parsed: {len(faith_scores)}/{len(records)}")
    print(f"Avg Relevancy:    {avg_rel:.2f} | Pass rate: {rel_pass_rate:.0%} | parsed: {len(rel_scores)}/{len(records)}")

    with open("/workspace/cuad_eval_results.json", "w") as f:
        json.dump(records, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
