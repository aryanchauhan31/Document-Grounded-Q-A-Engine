# Document-Grounded Q&A Engine

An agentic, production-grade Retrieval-Augmented Generation system. A `ReActAgent` decides how to answer each question — retrieving and grounding its response through a self-correcting workflow, or reaching for a calculator tool — rather than following one hardcoded retrieve-then-generate path. The serving layer autoscales, load-balances across multiple GPUs, and evaluates itself against a real legal-contract corpus using a second, independent LLM as judge.

This started as a single-notebook RAG baseline (`engine.py`, `chatbot_rag.py`) and was rebuilt in stages into the system described below. Both the original files and the production stack are kept in this repo — see [Files](#files) for what's what.

## Why "agentic," specifically

Most of what gets called an "agentic RAG system" is really a fixed pipeline: retrieve, stuff into a prompt, generate, return. Nothing here decides anything. This project is built around three places where the system actually makes a decision instead of following a script:

1. **Tool routing.** `rag_serve.py`'s `ReActAgent` is hand a `grounded_qa` tool and a `calculator` tool and reasons (via the ReAct Thought/Action/Observation loop) about which one a question needs — it isn't told in advance. Ask it arithmetic and it reaches for the calculator; ask it a contract question and it retrieves.
2. **Self-correction.** `grounded_qa` isn't a single retrieve-and-generate call — it's a `Workflow` (`RAGWorkflow`) that generates an answer, runs a `FaithfulnessEvaluator` against its own output, and if that check fails, *widens retrieval and tries again* (up to `MAX_RETRIES`), autonomously, before ever returning to the agent. The system checks its own work and repairs it without being asked to.
3. **Capacity decisions.** The serving layer (Ray Serve) autoscales the orchestration tier up and down based on live request load — nobody sets the replica count; it's a runtime decision made from `target_ongoing_requests`.

## Architecture

```
client
  -> Ray Serve HTTP proxy (:8000)              [load balancer + autoscaler, 1-4 replicas]
       -> RAGService replica                    [stateless; round-robins across backends below]
            -> ReActAgent                       [decides: grounded_qa tool, or calculator tool]
                 -> RAGWorkflow                  [retrieve -> generate -> faithfulness check -> retry]
                      -> hybrid retriever         [Pinecone dense (cosine, bge-small-en-v1.5) + local BM25 sparse]
                      -> vLLM backend (GPU 0 or 1) [zephyr-7b-alpha, OpenAI-compatible API]
```

Two independent vLLM instances (`vllm`, `vllm2`), one per GPU, form a fixed backend pool. Ray Serve replicas are stateless and scale independently of that pool -- more replicas doesn't mean more GPUs, it means more concurrent orchestration capacity sharing the same fixed generation capacity underneath.

## Corpus: CUAD (real legal contracts, not a toy example)

The retrieval corpus is the [CUAD dataset](https://huggingface.co/datasets/chenghao/cuad_qa) — 408 real commercial contracts (~11,400 chunks after splitting), indexed into Pinecone (`cuad-legal-rag`, 384-dim, cosine). `ingest_cuad.py` is the one-off batch job that does the chunk/embed/upsert; `rag_serve.py` queries the already-populated index at request time, it never re-embeds the corpus.

Dense retrieval goes through Pinecone; BM25 (sparse) is rebuilt locally from the same source text on each replica startup, since BM25 needs raw node text and Pinecone only stores vectors. The two legs are fused with `QueryFusionRetriever` (reciprocal rank fusion).

## LLM-as-judge evaluation

Evaluation uses a **separate model as judge**, deliberately not the model being evaluated:

- `zephyr-7b-alpha` (GPU 0) generates the answers under test.
- `Qwen2.5-14B-Instruct` (a dedicated `judge` service, temporarily borrowing GPU 1 from `vllm2`) grades them via `FaithfulnessEvaluator` and `AnswerRelevancyEvaluator`.

This isn't cosmetic. The first pass at this eval used `zephyr-7b-alpha` as its own judge, and `AnswerRelevancyEvaluator` came back with every single relevancy score unparseable (`None`) — the 7B model couldn't reliably produce the structured `[RESULT] <score>` output the evaluator's regex parser expects. Swapping in the larger, more instruction-compliant Qwen model fixed that completely (15/15 parsed) and also *changed* several individual faithfulness verdicts, meaning the two models didn't even agree on the same answers — a real reminder that an LLM judge's output is itself a claim, not ground truth.

Run it: `eval_cuad.py` (needs the `judge` service up and `vllm2` stopped — see `deploy/judge.sh`'s note on GPU sharing).

## What moved the needle on eval scores

Two problems, found by actually reading a full eval run rather than trusting the aggregate number:

1. **The questions weren't questions.** CUAD's raw fields are category labels ("Document Name", "Parties")fed straight into the pipeline as if they were natural-language questions. `rebuild_eval_questions.py` rephrases them ("What is the name of this document?") and, just as importantly, samples a *diverse* mix of the ~40 CUAD categories per contract instead of always the first one in the dataset's row order (which happened to always be "Document Name" — the first eval run's 15 questions were, without anyone intending it, all the same question).
2. **The answers weren't answers.** `response_mode="compact"` was returning bare extracted phrases. A custom `text_qa_template` (`QA_TEMPLATE` in `rag_serve.py`/`eval_cuad.py`) instructs the synthesizer to answer in a complete sentence that restates the question.

Faithfulness went 53% -> 87%; relevancy went from unparseable to 14/15 full marks. Neither fix touched the model, the retriever, or the judge — the pipeline's own inputs were the bug.

## Real infra bugs hit along the way (and why the fixes look the way they do)

- **`torch`/CUDA driver mismatch (`RuntimeError: Engine core initialization failed`).** `pip install vllm` resolved a `torch` build targeting CUDA 13, but this box's driver only advertised CUDA 12.8. Fixed by installing `cuda-compat-13-0` (NVIDIA's official forward-compatibility shim) and pointing `LD_LIBRARY_PATH` at it — not by downgrading torch, which would have fought vLLM's own compiled kernels (see next point).
- **vLLM's own kernels need a matching CUDA runtime independent of torch.** After the driver fix, a *different* import (`libcudart.so.13: cannot open shared object file`) turned out to need `nvidia/cu13/lib` (torch's own bundled runtime, not the system one) on `LD_LIBRARY_PATH` explicitly — dynamic loading via `ctypes.CDLL` doesn't get the same RPATH resolution torch's own extensions get.
- **`cuda-compat-13-0`'s install silently repointed `/usr/local/cuda` at an incomplete toolkit** (compat libs only, no `nvcc`). This caused flashinfer's JIT kernel compilation to fail with `nvcc: not found`, which manifested as both vLLM replicas cleanly exiting (`exit 0`) about 56 seconds after every restart — and because `autorestart=unexpected` (supervisor's default) treats exit-0 as an *intentional* stop, it never retried. Two separate fixes: repoint the symlink at the complete CUDA 12.8 toolkit (still fully compatible with the L40S's `sm_89` arch), and set `autorestart=true` so the service self-heals regardless of exit code.
- **Ray Serve can't pickle a deployment class that closes over already-constructed LLM/agent objects.** The first `rag_serve.py` built the `ReActAgent` and its LLM client at module level; Ray's replica-shipping mechanism tries to `cloudpickle` the whole class including that closure, and failed on a non-serializable object deep inside the OpenAI client. Fix: construct everything inside `RAGService.__init__`, which runs per-replica after the actor already exists — nothing gets pickled, only the class definition does.
- **`QueryFusionRetriever` silently falls back to a global `Settings.llm`** if you don't pass `llm=` explicitly, which isn't set in a Ray Serve worker process — this surfaced as an attempt to call the real OpenAI API and fail on a missing key. Always pass `llm=` explicitly per-backend when running multiple LLM instances in one process.
- **`AnswerRelevancyEvaluator` never sets `.passing`** on the result it returns (only `.score`) — unlike `FaithfulnessEvaluator`, which does. Code that reports a "pass rate" from `.passing` for both evaluators will silently show 0% relevancy pass rate forever, regardless of how good the actual scores are. The real signal is `.score`, thresholded manually against `AnswerRelevancyEvaluator`'s own `score_threshold` (default 2.0, since its raw judge score is 0-2 before normalization).

## Scaling: what the numbers actually showed

Load-testing `loadtest.py` against progressively higher concurrency (single vLLM backend, before the dual-GPU + Ray Serve setup) found the real throughput ceiling at concurrency ~8 (~0.7 req/s), not a number anyone guessed — past that, throughput flatlined while p50 latency climbed linearly (queueing, not failure; zero errors at every level tested up to 32 concurrent). Adding a second GPU-bound vLLM replica plus Ray Serve's autoscaling (1->4 orchestration replicas) pushed the ceiling to ~1.05-1.09 req/s — a real ~1.5x, short of a clean 2x because each Serve replica's backend round-robin counter starts independently, so early requests across newly-spawned replicas can pile onto the same backend before it evens out.

## Files

| File | What it is |
|---|---|
| `rag_serve.py` | Production entrypoint: Ray Serve autoscaling deployment, hybrid Pinecone+BM25 retrieval, `ReActAgent` with `grounded_qa`/`calculator` tools, faithfulness-gated `RAGWorkflow`. |
| `ingest_cuad.py` | One-off batch job: load CUAD, chunk, embed, upsert into Pinecone. Also samples a raw QA eval set. |
| `rebuild_eval_questions.py` | Rebuilds the eval question set with natural-language phrasing and category diversity (see [above](#what-moved-the-needle-on-eval-scores)). |
| `eval_cuad.py` | Runs the faithfulness/relevancy eval against real CUAD questions, using a separate judge LLM. |
| `loadtest.py` | Async concurrency ramp load test against the `/rag` endpoint; reports p50/p95/p99 latency and throughput per concurrency level. |
| `deploy/` | Supervisor wrapper scripts + config for the four backing services (`vllm`, `vllm2`, `judge`, `rayserve`), written for a Vast.ai-style supervisor+Caddy box. Adapt paths/venv activation for other environments. |
| `engine.py` | Original single-process baseline: vLLM (in-process) + hybrid retrieval + faithfulness-retry workflow + tool-using agent, no serving layer. Still runnable standalone. |
| `chatbot_rag.py` | Earliest baseline: `HuggingFaceLLM` + a single chat-engine call, no evaluation. |
| `questions.json` / `answers.json` | The original toy benchmark set (unrelated to the CUAD eval questions, which live only on the deployment box as `cuad_eval_questions.json`). |

## Running it

Needs 2 GPUs (developed against 2x NVIDIA L40S, 46GB each), a Pinecone API key (`PINECONE_API_KEY` env var), and the packages in `requirements.txt`.

```bash
pip install -r requirements.txt
python3 ingest_cuad.py                    # one-time: populate the Pinecone index
python3 rebuild_eval_questions.py         # one-time: build a diverse natural-language eval set

# bring up the two generation backends + the Ray Serve orchestration layer
# (see deploy/*.sh for the exact env vars each needs, notably LD_LIBRARY_PATH
# for the forward-compat/nvcc fixes above)
vllm serve HuggingFaceH4/zephyr-7b-alpha --port 18000 &
vllm serve HuggingFaceH4/zephyr-7b-alpha --port 18002 &   # CUDA_VISIBLE_DEVICES=1
serve run rag_serve:rag_app

curl -X POST http://127.0.0.1:8000/rag   -d '{"question": "What is the governing law of this contract?"}'
curl -X POST http://127.0.0.1:8000/query -d '{"question": "What is 94 plus 89?"}'   # routes to calculator
```

To re-run the eval (stop `vllm2`, it shares GPU 1 with the judge model):
```bash
supervisorctl stop vllm2 && supervisorctl start judge
python3 eval_cuad.py
supervisorctl stop judge && supervisorctl start vllm2   # restore full serving capacity
```

## Next steps

- Fix the round-robin imbalance noted in [Scaling](#scaling-what-the-numbers-actually-showed) with a shared backend-selection counter instead of a per-replica-local one.
- Chunk legal text on clause/section boundaries instead of a fixed 512-token window, which currently can cut a clause mid-sentence.
- A stronger generation model would likely improve both faithfulness and answer phrasing quality beyond what prompt-level fixes alone can do.
