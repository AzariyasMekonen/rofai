from pathlib import Path

from app.ingest import MAX_CHARS, OVERLAP_CHARS, build_chunks, chunk_paragraphs, clean_text


def test_clean_text_dehyphenates_and_drops_debris():
    raw = "The school was found-\ned in 1986 by\nour founder.\n\n12\n~~ |\n\nSecond para-\ngraph here."
    paragraphs = clean_text(raw)
    assert paragraphs == ["The school was founded in 1986 by our founder.", "Second paragraph here."]


def test_clean_text_keeps_ethiopic():
    paragraphs = clean_text("ትምህርት ቤቱ በ1978 ተመሠረተ።")
    assert paragraphs == ["ትምህርት ቤቱ በ1978 ተመሠረተ።"]


def test_chunks_respect_max_size_and_overlap():
    paragraphs = [f"Paragraph {i} " + "word " * 60 for i in range(30)]
    chunks = chunk_paragraphs(paragraphs)
    assert len(chunks) > 1
    assert all(len(c) <= MAX_CHARS for c in chunks)
    # The end of one chunk is repeated at the start of the next
    assert chunks[0][-60:] in chunks[1][: OVERLAP_CHARS + 60]


def test_short_paragraphs_overlap_whole():
    paragraphs = [f"Short paragraph number {i} about the school." for i in range(80)]
    chunks = chunk_paragraphs(paragraphs)
    assert len(chunks) > 1
    assert chunks[0].split("\n")[-1] in chunks[1].split("\n")


def test_long_paragraph_is_split():
    paragraph = "This is a sentence about the school. " * 200
    chunks = chunk_paragraphs([paragraph])
    assert len(chunks) > 1
    assert all(len(c) <= MAX_CHARS for c in chunks)


def test_build_chunks_marks_facts_as_priority(tmp_path: Path):
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "brochure.txt").write_text(
        "The library opened in 1995 with donated books from parents and teachers.", encoding="utf-8"
    )
    facts = tmp_path / "facts.md"
    facts.write_text("The school was founded by Tsehay Roschli and opened its doors to its first students.", encoding="utf-8")

    chunks = build_chunks(sources, facts)

    by_source = {c.source: c for c in chunks}
    assert by_source["facts.md"].priority is True
    assert by_source["brochure.txt"].priority is False
    assert by_source["brochure.txt"].years == [1995]
