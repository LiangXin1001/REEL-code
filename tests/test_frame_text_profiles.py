"""Public-release subset of the internal frame-text profile tests.

Only tests that exercise the published runner with synthetic temporary data
belong here.  Internal recovery wrappers, SLURM files, and licensed benchmark
manifests are intentionally excluded from the public repository.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from scripts import run_controlled_memory_policy_cgbench as runner


def test_balanced24_is_the_existing_formal_prompt() -> None:
    hidden, context = runner.frame_level_write_prompts("balanced24")
    assert hidden == runner.FRAME_LEVEL_WRITE_PROMPT
    assert "at most 24 words total" in hidden
    assert hidden.startswith("Questions are hidden.")
    assert not context.startswith("Questions are hidden.")


def test_answer_schema_uses_one_concrete_label() -> None:
    assert '{"answer":"A"' in runner.ANSWER_PROMPT
    assert '{"answer":"A|B|C' not in runner.ANSWER_PROMPT
    assert "exactly one uppercase" in runner.ANSWER_PROMPT


def test_formal_64_frame_answer_prompt_hash_is_locked() -> None:
    digest = hashlib.sha256(runner.answer_prompt_for_frames(64).encode()).hexdigest()
    assert digest == "baa64a3423f0e47e457b01e87fb8739d924424f7d9e8bcf60a0c508238e27ae9"


def test_minimal_answer_repair_schema_is_short_and_choice_bounded() -> None:
    response_format = runner.minimal_answer_repair_response_format(
        ["A", "B", "C", "D"]
    )
    wrapped = response_format["json_schema"]
    schema = wrapped["schema"]
    assert response_format["type"] == "json_schema"
    assert wrapped["strict"] is True
    assert schema["additionalProperties"] is False
    assert schema["properties"]["answer"]["enum"] == ["A", "B", "C", "D"]
    assert schema["properties"]["evidence_spans"]["maxItems"] == 0
    assert schema["required"] == ["answer", "evidence_spans", "rationale"]


def test_capacity_and_detail_profiles_preserve_the_formal_schema() -> None:
    balanced, _ = runner.frame_level_write_prompts("balanced24")
    expanded, _ = runner.frame_level_write_prompts("expanded48")
    detail, _ = runner.frame_level_write_prompts("detail24")
    schema = "Frame <index> | entities: <visible people and objects> | action/state:"
    assert schema in balanced and schema in expanded and schema in detail
    assert "at most 24 words total" in balanced
    assert "write 32 to 48 words total" in expanded
    assert "prioritize object contents" not in expanded
    assert "at most 24 words total" in detail
    assert "prioritize object contents" in detail
    assert "Questions are hidden" in detail


def test_prebuilt_repository_resolves_sharded_layout(tmp_path) -> None:
    repository = tmp_path / "shard_2" / "run" / "video-1.repository.txt"
    repository.parent.mkdir(parents=True)
    repository.write_text(
        "Frame 0 | entities: cup | action/state: still | visible text: none\n"
    )
    assert runner.resolve_prebuilt_repository(tmp_path, "video-1") == repository


def test_derived_profiles_make_the_capacity_comparison_explicit() -> None:
    fixed = runner.frame_text_derived_spec("hybrid_fixed")
    added = runner.frame_text_derived_spec("hybrid_added")
    compact = runner.frame_text_derived_spec("compact_only")

    assert fixed is not None and fixed["global_summary"] is True
    assert fixed["compact_frame_table"] is True
    assert added is not None and added["global_summary"] is True
    assert added["compact_frame_table"] is False
    assert compact is not None and compact["global_summary"] is False
    assert compact["compact_frame_table"] is True
    assert runner.frame_text_derived_spec("balanced24") is None


def test_global_summary_is_one_bounded_line() -> None:
    value = runner.normalize_global_summary(
        "Global summary |  A person opens   a red box.  ", 9
    )
    assert value == "Global summary | A person opens a red box."

    with pytest.raises(ValueError, match="exactly one physical line"):
        runner.normalize_global_summary(
            "Global summary | first\nGlobal summary | second", 20
        )
    with pytest.raises(ValueError, match="word limit"):
        runner.normalize_global_summary("Global summary | one two three", 4)
    with pytest.raises(ValueError, match="UTF-8 byte limit"):
        runner.normalize_global_summary("Global summary | " + "界" * 400, 500)


def test_compact_frame_table_round_trip_preserves_all_fact_fields() -> None:
    original = "\n".join(
        [
            (
                r"Frame 0 | entities: red box \\ label | action/state: opens | "
                r"visible text: A | B"
            ),
            "Frame 1 | entities: person | walking left | none",
            "Frame 2 | entities: sign | static | EXIT |",
            "Frame 3 | an unusual unlabelled payload",
        ]
    )
    expected = [0, 1, 2, 3]

    compact, records = runner.compact_frame_table(original, expected)

    assert runner.parse_compact_frame_table(compact, expected) == records
    assert [record[0] for record in records] == ["F", "U", "UT", "R"]
    assert records == runner.parse_standard_frame_fields(original, expected)
    restored = "\n".join(runner.restore_source_frame_line(row) for row in records)
    assert runner.parse_standard_frame_fields(restored, expected) == records


def test_compact_escape_handles_empty_tabs_and_backslashes() -> None:
    for value in ("", "plain", "a\tb", r"a\\b", r"literal \\t and \\e"):
        assert runner.compact_unescape(runner.compact_escape(value)) == value


def test_timestamped_compact_repository_keeps_summary_and_times() -> None:
    original = "\n".join(
        [
            "Frame 0 | entities: cup | action/state: still | visible text: none",
            "Frame 1 | entities: hand | action/state: lifts cup | visible text: none",
        ]
    )
    compact, _ = runner.compact_frame_table(original, [0, 1])
    repository = "Global summary | A hand lifts a cup.\n" + compact

    timed = runner.attach_frame_times(
        repository,
        [0.125, 2.5],
        require_global_summary=True,
        summary_word_limit=20,
        compact_frame_table_required=True,
    )

    assert timed.startswith("Global summary | A hand lifts a cup.\n")
    assert "F\t0\t0.125s\tcup\tstill\tnone" in timed
    assert "F\t1\t2.500s\thand\tlifts cup\tnone" in timed


def test_summary_and_timestamped_repository_resolve_sharded_layout(tmp_path) -> None:
    run_dir = tmp_path / "shard_3" / "run"
    run_dir.mkdir(parents=True)
    summary = run_dir / "video-1.global_summary.txt"
    timestamped = run_dir / "video-1.repository_with_time.txt"
    summary.write_text("Global summary | A cup is visible.\n")
    timestamped.write_text("Frame 0 | time: 0.000s | entities: cup\n")

    assert runner.resolve_prebuilt_global_summary(tmp_path, "video-1") == summary
    assert (
        runner.resolve_prebuilt_timestamped_repository(tmp_path, "video-1")
        == timestamped
    )


def test_imported_writer_cost_accepts_legacy_ids_but_normalizes_them(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    rows = []
    for index, stamp in enumerate((0.0, 2.5)):
        rows.append(
            {
                "run_id": "old",
                "git_commit": "abc",
                "method_id": "old-text",
                "dataset_version": "data",
                "video_id": "video-1",
                "question_id": "__video_write__",
                "phase": "write",
                "model_id": "model",
                "cache_status": "miss",
                "input_text_tokens": 1,
                "output_text_tokens": 1,
                "visual_tokens": 1,
                "decoded_frames": 0,
                "repeated_frames": 0,
                "unique_frame_ids": [f"video-1:controlled:{index}:{stamp:.6f}"],
                "selected_times": [stamp],
                "raw_video_seconds_read": 0.0,
                "latency_s": 1.0,
                "gpu_count": 1,
                "allocated_gpu_count": 1,
                "gpu_seconds": 1.0,
                "peak_vram_bytes": 1,
                "api_cost_usd": 0.0,
                "error": "",
                "measurement_note": "Question-blind frame-level text write: test",
            }
        )
    (source / "cost_ledger.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    current_ids = [
        "video-1:controlled:abc:native:0:0.000000",
        "video-1:controlled:abc:native:1:2.500000",
    ]

    runner.import_prebuilt_frame_text_writer_cost(
        directory=source,
        ledger_path=tmp_path / "imported.jsonl",
        run_id="new",
        commit="def",
        method="new-text",
        dataset_version="data",
        model_id="model",
        video_id="video-1",
        frame_ids=current_ids,
        selected_times=[0.0, 2.5],
        selected_times_sha256="0" * 64,
        timestamp_manifest_sha256="",
        out_dir=tmp_path,
    )

    imported = [
        json.loads(line)
        for line in (tmp_path / "imported.jsonl").read_text().splitlines()
    ]
    assert [row["unique_frame_ids"][0] for row in imported] == current_ids
    assert all(row["canonical_text_cost_imported"] is True for row in imported)


def test_legacy_balanced_acceptance_is_identified_by_exact_prompt(tmp_path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    payload = {
        "accepted": True,
        "source_frames_per_video": 64,
        "writer_chunk_size": 8,
        "frame_writer_tokens": 1024,
        "prompt_sha256": runner.hashlib.sha256(
            runner.FRAME_LEVEL_WRITE_PROMPT.encode()
        ).hexdigest(),
    }
    path = run / "frame_text_schema_acceptance.json"
    path.write_text(json.dumps(payload))

    loaded, observed_path = runner.load_base_frame_text_acceptance(tmp_path)

    assert observed_path == path
    assert loaded["frame_text_profile"] == "balanced24"
    assert loaded["question_context"] == "hidden"
    assert loaded["legacy_missing_fields_inferred_from_exact_prompt"] is True
