# RAG evaluation report

Eval set: 40 questions (35 answerable, 5 unanswerable). Corpus: 26 documents.

## Results

| Config | Chunk/overlap | Retriever | k | Generator | Recall@k | MRR | Fact recall | Accuracy | Grounded sent. | Halluc. answers | Refusal acc. | p50 ms | p95 ms | Est. $/1k q |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 01-baseline | 120/30 | bm25 | 4 | mock-extractive | 1.000 | 0.986 | 0.857 | 0.857 | 1.000 | 0.000 | 0.900 | 0.19 | 0.24 | 1.500 |
| 02-small-chunks | 30/5 | bm25 | 4 | mock-extractive | 1.000 | 1.000 | 0.786 | 0.771 | 1.000 | 0.000 | 0.875 | 0.10 | 0.11 | 0.825 |
| 03-hybrid-rerank | 120/30 | hybrid a=0.5 + rerank | 4 | mock-extractive | 1.000 | 0.986 | 0.857 | 0.857 | 1.000 | 0.000 | 0.900 | 0.66 | 0.74 | 1.500 |
| 04-k1 | 120/30 | bm25 | 1 | mock-extractive | 0.943 | 0.971 | 0.829 | 0.829 | 1.000 | 0.000 | 0.875 | 0.06 | 0.08 | 0.735 |
| 05-hallucination-probe | 120/30 | bm25 | 4 | mock-hallucinating | 1.000 | 0.986 | 0.714 | 0.714 | 0.714 | 0.452 | 0.900 | 0.20 | 0.25 | 1.535 |

Recall@k and MRR use gold documents. Fact recall and accuracy cover answerable questions only. Grounded sent. is the share of answer sentences supported by the chunk they cite. Halluc. answers is the share of non-refused answers with at least one unsupported sentence. Refusal acc. counts a refusal as correct only for unanswerable questions. Est. $/1k q projects the token count onto each config's cost model.

## Configurations

- **01-baseline**: BM25 over 120-word chunks, top 4. Chunks: 26. Cost model: `claude-sonnet-5-5`. Unanswerable refused: 5/5. False refusals: 4. Avg tokens in/out: 607/29.
- **02-small-chunks**: BM25 over 30-word chunks, top 4. Tests whether facts survive tight chunk boundaries. Chunks: 91. Cost model: `claude-sonnet-5-5`. Unanswerable refused: 5/5. False refusals: 5. Avg tokens in/out: 296/23.
- **03-hybrid-rerank**: BM25 plus hashing vectors (alpha 0.5), reranked, top 4. Chunks: 26. Cost model: `claude-sonnet-5-5`. Unanswerable refused: 5/5. False refusals: 4. Avg tokens in/out: 607/29.
- **04-k1**: BM25 over 120-word chunks, top 1. Cheapest prompt. Chunks: 26. Cost model: `claude-sonnet-5-5`. Unanswerable refused: 5/5. False refusals: 5. Avg tokens in/out: 231/27.
- **05-hallucination-probe**: Baseline retrieval with a mock that corrupts about 30% of answers. Checks that the grounding metric catches them. Chunks: 26. Cost model: `claude-sonnet-5-5`. Unanswerable refused: 5/5. False refusals: 4. Avg tokens in/out: 607/32.

## Failure examples

### 01-baseline

- Wrong refusal (q09): Does Acme give back the processing fee when I refund a charge?
- Wrong refusal (q17): What is the largest amount allowed for one instant payout?
- Missed fact (q25): What is the default page size for list endpoints?
- Wrong refusal (q27): For how long does Acme retry a failed webhook delivery?
- Wrong refusal (q32): How long is the longest free trial for a subscription?

### 02-small-chunks

- Wrong refusal (q09): Does Acme give back the processing fee when I refund a charge?
- Missed fact (q16): What do instant payouts cost?
- Wrong refusal (q17): What is the largest amount allowed for one instant payout?
- Wrong refusal (q24): What is the maximum length of an idempotency key?
- Missed fact (q25): What is the default page size for list endpoints?

### 03-hybrid-rerank

- Wrong refusal (q09): Does Acme give back the processing fee when I refund a charge?
- Wrong refusal (q17): What is the largest amount allowed for one instant payout?
- Missed fact (q25): What is the default page size for list endpoints?
- Wrong refusal (q27): For how long does Acme retry a failed webhook delivery?
- Wrong refusal (q32): How long is the longest free trial for a subscription?

### 04-k1

- Wrong refusal (q09): Does Acme give back the processing fee when I refund a charge?
- Wrong refusal (q17): What is the largest amount allowed for one instant payout?
- Missed fact (q25): What is the default page size for list endpoints?
- Wrong refusal (q27): For how long does Acme retry a failed webhook delivery?
- Wrong refusal (q31): What is the minimum TLS version for API traffic?

### 05-hallucination-probe

- Unsupported (q02): "International cards cost an additional 2.5% on top of the domestic rate. [pricing-standard#c0]" (numbers not in cited text: 2.5)
- Unsupported (q04): "Acme waives this fee for registered nonprofit organizations. [pricing-ach#c0]" (low term support 0.33)
- Unsupported (q06): "The Growth plan costs $50 per month and includes 10 team seats, custom checkout branding, and priority email support. [pricing-plans#c0]" (numbers not in cited text: 50)
- Unsupported (q08): "Acme waives this fee for registered nonprofit organizations. [refunds-policy#c0]" (low term support 0.33)
- Unsupported (q13): "Acme waives this fee for registered nonprofit organizations. [disputes#c0]" (low term support 0.33)
- Missed fact (q02): How much extra do international cards cost?
- Missed fact (q06): How much does the Growth plan cost per month?
- Wrong refusal (q09): Does Acme give back the processing fee when I refund a charge?
- Missed fact (q15): When does my first payout arrive?
- Wrong refusal (q17): What is the largest amount allowed for one instant payout?
