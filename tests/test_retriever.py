from rageval.corpus import Chunk, chunk_corpus, load_corpus
from rageval.embedding import HashingEmbedder, cosine
from rageval.retriever import Hit, RetrievalConfig, Retriever, rerank


def test_hashing_embedding_is_stable_and_normalized() -> None:
    emb = HashingEmbedder(dim=64)
    a, b = emb.embed("instant payout fee"), emb.embed("instant payout fee")
    assert a == b
    assert abs(cosine(a, a) - 1.0) < 1e-9


def test_rerank_promotes_full_coverage() -> None:
    hits = [
        Hit(Chunk("a#c0", "a", "payout schedule is weekly"), 2.0),
        Hit(Chunk("b#c0", "b", "instant payout fee is one percent"), 1.9),
        Hit(Chunk("c#c0", "c", "webhook retry"), 0.0),
    ]
    reranked = rerank("instant payout fee", hits, idf=lambda t: 1.0)
    assert reranked[0].chunk.chunk_id == "b#c0"


def test_bm25_and_hybrid_find_the_gold_document() -> None:
    chunks = chunk_corpus(load_corpus("data/corpus"), 120, 30)
    for mode in ("bm25", "hybrid"):
        retriever = Retriever(chunks, RetrievalConfig(mode=mode, k=3, rerank=True))
        hits = retriever.retrieve("How much is the dispute fee?")
        assert len(hits) == 3
        assert hits[0].chunk.doc_id == "disputes"
