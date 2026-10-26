from pestate.vlm_parse import (
    extract_cited_frame_numbers,
    normalized_evidence_sufficiency,
    parse_answer,
    parse_evidence_response,
)


def test_parse_answer_from_jsonish_response():
    raw = 'prefix {"answer": "c", "evidence_sufficiency": "sufficient"} suffix'
    assert parse_answer(raw) == "C"


def test_multi_answer_json_is_not_salvaged_as_first_label():
    raw = '{"answer": "A|B|C", "evidence_spans": [], "rationale": "unsure"}'
    assert parse_answer(raw) == ""
    parsed = parse_evidence_response(raw)
    assert parsed.answer == ""
    assert parsed.parse_error == "invalid_answer"


def test_extract_cited_frame_numbers_deduplicates_common_formats():
    raw = "Frame 0 shows the setup. frame: 2 shows the action. Frame-2 is repeated."
    assert extract_cited_frame_numbers(raw) == [0, 2]


def test_extract_cited_frame_numbers_reads_json_frame_values():
    raw = '{"relevant_moments": [{"frame": "0"}, {"frame": "Frame 3"}]}'
    assert extract_cited_frame_numbers(raw) == [0, 3]


def test_normalized_evidence_sufficiency_reads_raw_response_json():
    row = {"raw_response": '{"answer": "A", "evidence_sufficiency": "Insufficient"}'}
    assert normalized_evidence_sufficiency(row) == "insufficient"


def test_parse_v3_evidence_spans_and_key_frames():
    raw = """{
      "answer": "D",
      "confidence": 0.7,
      "evidence_sufficiency": "sufficient",
      "need_more_evidence": false,
      "evidence_spans": [
        {"role": "transition", "start_frame": 2, "end_frame": 4, "key_frames": [2, 4]}
      ],
      "missing_evidence": ["before"],
      "rationale": "visible action"
    }"""
    parsed = parse_evidence_response(raw)
    assert parsed.answer == "D"
    assert parsed.confidence == 0.7
    assert parsed.evidence_sufficiency == "sufficient"
    assert parsed.need_more_evidence is False
    assert parsed.evidence_spans[0]["role"] == "transition"
    assert parsed.missing_evidence == ("before",)
    assert extract_cited_frame_numbers(raw) == [2, 4]
