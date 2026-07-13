import json
import os
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, AnswerRelevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

with open("eval_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

dataset = Dataset.from_list(data)

raw_llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL"),
    api_key=os.getenv("GOOGLE_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    model_kwargs={"response_format": {"type": "json_object"}},
)

raw_embeddings = OpenAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    check_embedding_ctx_length=False,
    encoding_format="float",
)

ragas_llm = LangchainLLMWrapper(raw_llm)
ragas_embeddings = LangchainEmbeddingsWrapper(raw_embeddings)

answer_relevancy_metric = AnswerRelevancy(strictness=1)
answer_relevancy_metric.llm = ragas_llm
answer_relevancy_metric.embeddings = ragas_embeddings

print(f"Evaluating {len(data)} questions...")

scores = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy_metric, context_precision, context_recall],
    llm=ragas_llm,
    embeddings=ragas_embeddings,
)

print(scores)

df = scores.to_pandas()
df.to_csv("ragas_results.csv", index=False)
print("\nSaved per-question breakdown to ragas_results.csv")