from rageval.corpus import chunk_corpus, load_corpus
from rageval.generation import REFUSAL, ExtractiveMockGenerator, LLMGenerator, citations, is_refusal
from rageval.llm import Completion
from rageval.retriever import RetrievalConfig, Retriever


def _setup() -> tuple[Retriever, ExtractiveMockGenerator]:
    retriever = Retriever(chunk_corpus(load_corpus("data/corpus"), 120, 30), RetrievalConfig(k=4))
    return retriever, ExtractiveMockGenerator(retriever.bm25.idf)


def test_is_refusal_variants() -> None:
    assert is_refusal(REFUSAL)
    assert is_refusal("I don’t know.")
    assert is_refusal("")
    assert not is_refusal("The fee is $15. [disputes#c0]")


def test_mock_refuses_off_corpus_question() -> None:
    retriever, gen = _setup()
    question = "Does Acme integrate natively with Shopify?"
    answer = gen.answer(question, [h.chunk for h in retriever.retrieve(question)])
    assert answer.text == REFUSAL


def test_mock_answers_with_citations_from_retrieved_chunks() -> None:
    retriever, gen = _setup()
    question = "How much is the dispute fee?"
    chunks = [h.chunk for h in retriever.retrieve(question)]
    answer = gen.answer(question, chunks)
    assert not is_refusal(answer.text)
    assert "$15" in answer.text
    assert set(citations(answer.text)) <= {c.chunk_id for c in chunks}


def test_mock_refuses_with_no_context() -> None:
    _, gen = _setup()
    assert gen.answer("How much is the dispute fee?", []).text == REFUSAL


class _RefusingModel:
    name = "fake"

    def complete(self, system: str, user: str) -> Completion:
        return Completion("", 10, 0, refused_by_provider=True)


def test_provider_refusal_maps_to_refusal_text() -> None:
    answer = LLMGenerator(_RefusingModel()).answer("q", [])
    assert answer.text == REFUSAL
    assert not answer.usage_estimated
