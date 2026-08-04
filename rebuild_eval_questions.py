"""
Rebuild /workspace/cuad_eval_questions.json with:
- a diverse mix of CUAD question categories per contract (not always
  "Document Name", which is what the dataset's row ordering defaulted to)
- natural-language question phrasing instead of raw CUAD category labels,
  so the AnswerRelevancyEvaluator judge is scoring a real question/answer
  pair instead of a bare category name against a bare extracted phrase.
"""
import json
import random

from datasets import load_dataset

N_SAMPLE = 50
random.seed(0)

CATEGORY_QUESTIONS = {
    "Document Name": "What is the name of this document?",
    "Parties": "Who are the parties to this contract?",
    "Agreement Date": "What is the agreement date of this contract?",
    "Effective Date": "What is the effective date of this contract?",
    "Expiration Date": "What is the expiration date of this contract?",
    "Renewal Term": "What is the renewal term of this contract?",
    "Notice Period To Terminate Renewal": "What notice period is required to terminate renewal of this contract?",
    "Governing Law": "What is the governing law of this contract?",
    "Most Favored Nation": "Does this contract include a most favored nation clause?",
    "Non-Compete": "Does this contract include a non-compete clause?",
    "Exclusivity": "Does this contract include an exclusivity clause?",
    "No-Solicit Of Customers": "Does this contract include a non-solicitation of customers clause?",
    "Competitive Restriction Exception": "Does this contract include a competitive restriction exception?",
    "No-Solicit Of Employees": "Does this contract include a non-solicitation of employees clause?",
    "Non-Disparagement": "Does this contract include a non-disparagement clause?",
    "Termination For Convenience": "Does this contract allow termination for convenience?",
    "Rofr/Rofo/Rofn": "Does this contract include a right of first refusal, offer, or negotiation?",
    "Change Of Control": "Does this contract include a change of control clause?",
    "Anti-Assignment": "Does this contract include an anti-assignment clause?",
    "Revenue/Profit Sharing": "Does this contract include a revenue or profit sharing arrangement?",
    "Price Restrictions": "Does this contract include price restrictions?",
    "Minimum Commitment": "Does this contract include a minimum commitment?",
    "Volume Restriction": "Does this contract include a volume restriction?",
    "IP Ownership Assignment": "Does this contract assign IP ownership?",
    "Joint IP Ownership": "Does this contract include joint IP ownership?",
    "License Grant": "What license is granted under this contract?",
    "Non-Transferable License": "Is the license granted under this contract non-transferable?",
    "Affiliate License-Licensor": "Does this contract grant a license to the licensor's affiliates?",
    "Affiliate License-Licensee": "Does this contract grant a license to the licensee's affiliates?",
    "Unlimited/All-You-Can-Eat-License": "Does this contract grant an unlimited or all-you-can-eat license?",
    "Irrevocable Or Perpetual License": "Does this contract grant an irrevocable or perpetual license?",
    "Source Code Escrow": "Does this contract include a source code escrow arrangement?",
    "Post-Termination Services": "Does this contract require post-termination services?",
    "Audit Rights": "Does this contract include audit rights?",
    "Uncapped Liability": "Does this contract include uncapped liability?",
    "Cap On Liability": "What is the cap on liability in this contract?",
    "Liquidated Damages": "Does this contract include liquidated damages?",
    "Warranty Duration": "What is the warranty duration in this contract?",
    "Insurance": "What insurance requirements does this contract include?",
    "Covenant Not To Sue": "Does this contract include a covenant not to sue?",
    "Third Party Beneficiary": "Does this contract include a third party beneficiary clause?",
}


def natural_question(category: str) -> str:
    return CATEGORY_QUESTIONS.get(category, f"What does this contract specify regarding {category}?")


print("Loading chenghao/cuad_qa ...")
ds = load_dataset("chenghao/cuad_qa", split="train")

by_title = {}
for row in ds:
    if row["answers"]["text"]:
        by_title.setdefault(row["title"], []).append(row)

titles = list(by_title.keys())
random.shuffle(titles)

eval_rows = []
for title in titles:
    row = random.choice(by_title[title])
    eval_rows.append({
        "question": f"{natural_question(row['question'])} (contract: {title})",
        "expected_answer": row["answers"]["text"][0],
        "category": row["question"],
        "title": title,
    })
    if len(eval_rows) >= N_SAMPLE:
        break

with open("/workspace/cuad_eval_questions.json", "w") as f:
    json.dump(eval_rows, f, indent=2)

cats = sorted(set(r["category"] for r in eval_rows))
print(f"Wrote {len(eval_rows)} eval questions spanning {len(cats)} categories: {cats}")
