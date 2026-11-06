import pytest

from rageval.corpus import Document, chunk_document, load_corpus, parse_markdown


def _doc(n_words: int) -> Document:
    return Document("d", "d", " ".join(f"w{i}" for i in range(n_words)))


def test_windows_have_fixed_size_and_overlap() -> None:
    chunks = chunk_document(_doc(10), size=4, overlap=1)
    assert [c.text for c in chunks] == ["w0 w1 w2 w3", "w3 w4 w5 w6", "w6 w7 w8 w9"]
    assert [c.chunk_id for c in chunks] == ["d#c0", "d#c1", "d#c2"]


def test_last_window_may_be_short() -> None:
    chunks = chunk_document(_doc(5), size=4, overlap=0)
    assert [c.text for c in chunks] == ["w0 w1 w2 w3", "w4"]


def test_short_document_gives_one_chunk() -> None:
    assert len(chunk_document(_doc(3), size=10, overlap=2)) == 1


def test_empty_document_gives_no_chunks() -> None:
    assert chunk_document(Document("e", "e", ""), size=4, overlap=1) == []


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (4, 4), (4, -1)])
def test_invalid_parameters_raise(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_document(_doc(5), size=size, overlap=overlap)


def test_markdown_markers_are_stripped() -> None:
    doc = parse_markdown("x", "# Title\n\nFirst line.\n- A bullet item.\n")
    assert doc.title == "Title"
    assert doc.text == "Title. First line. A bullet item."


def test_corpus_loads_all_documents() -> None:
    docs = load_corpus("data/corpus")
    assert len(docs) >= 25
    assert len({d.doc_id for d in docs}) == len(docs)
