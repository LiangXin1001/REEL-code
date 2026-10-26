# Data

REEL does not redistribute benchmark videos, subtitles, annotations, or
question-level manifests. Download each benchmark from its official source,
accept its terms, and keep the resulting files outside Git.

## Official sources

### CG-Bench

- Project and download instructions: <https://github.com/CG-Bench/CG-Bench>
- Gated Hugging Face dataset: <https://huggingface.co/datasets/CG-Bench/CG-Bench>

Access to the Hugging Face repository requires accepting the dataset
conditions. The benchmark contains links or copies derived from third-party
videos; follow the benchmark's conditions and the rights attached to each
original video. Access to CG-Bench is not permission to republish it here.

### HD-EPIC

- Project page: <https://hd-epic.github.io/site/>
- Official data record: <https://doi.org/10.5523/bris.3cqb5b81wk2dc2379fx1mrxh47>
- Download and evaluation tools: <https://github.com/hd-epic>

Use the official download tools and comply with the HD-EPIC license and access
terms. REEL does not mirror the download scripts, annotations, or media.

## Local layout and manifest schema

The runner does not require a hard-coded cluster layout. A practical local
layout is:

```text
reel-data/
  cgbench/videos/...
  hd-epic/videos/...
  manifests/cg_mq50.questions.jsonl
  manifests/hdepic_51.questions.jsonl
```

Keep this directory outside the repository. Pass its location explicitly to
the runner. A relative `video_path` in a question manifest is resolved from
`--workspace-root`; an absolute path works locally but must never be committed.

The controlled runner consumes one JSON object per question. Every row must
contain at least:

```json
{
  "video_uid": "stable-video-id",
  "video_path": "path/relative/to/workspace-root.mp4",
  "duration": 1800.0,
  "qid": "stable-question-id",
  "within_video_question_order": 0,
  "question": "...",
  "choices": ["...", "...", "...", "..."],
  "right_answer": "A"
}
```

`within_video_question_order` is an integer fixed before inference. The
canonical CG builder emits zero-based values; the HD builder emits one-based
values. The runner only sorts on this field, so preserve the builder's values
rather than renumbering them. Preserve the official question text, choice
order, answer, and identifier. The CG order is the deterministic seeded order
described below, not the source file's original order. A locally materialized
copy changes runtime path fields such as `video_path` and CG `subtitle_path`;
when those strings differ, its byte-level SHA-256 also differs from the
accepted-run manifest hash below.

## CG MQ-50 cohort

The paper cohort has exactly 50 videos and six ordered questions per video:
300 question rows in total.

The cohort was fixed before model inference with this procedure:

1. Reject videos with fewer than six questions or any missing required
   question field.
2. Require locally playable media and an official subtitle file. Subtitles are
   an eligibility check only; subtitle content is never supplied to any REEL
   path.
3. Assign each eligible video to a stratum defined by duration bin, majority
   high-level question type, and capped question count.
4. Use seed `20260722` and deterministic nested round-robin sampling across
   strata until 50 videos are selected.
5. For each selected video, initialize
   `random.Random(f"20260722:{video_uid}")`, shuffle that video's eligible
   questions, and retain the first six in that deterministic seeded order.

In the audited official snapshot, 300 videos passed the local media check and
114 of those also had official subtitles. The selected duration distribution
was:

| Duration | Videos |
| --- | ---: |
| 10--15 min | 10 |
| 15--20 min | 5 |
| 20--30 min | 10 |
| 30--45 min | 15 |
| 45--60 min | 9 |
| over 60 min | 1 |

The selected videos have minimum/mean/median/maximum durations of
10.02/29.98/29.89/69.00 minutes.

Audit identifiers:

- canonical 300-row questions manifest SHA-256:
  `1f9c3e8fd6f24c37a44c7848e320ff6be55a926574250cabf19fa20757a599d1`
- official source-annotation snapshot SHA-256:
  `f2db9d2e8144e30806c3eab3ea50e7e1681bf9dd31d0dd6ce2871f960559de12`

The canonical manifest contents are not redistributed. The public builder is
[`scripts/audit_cgbench_and_build_mq.py`](../scripts/audit_cgbench_and_build_mq.py).
Given the audited official annotations and media/subtitle roots, it reproduces
the selected IDs and their seeded question order. Users must check the source
snapshot SHA and cohort counts before inference.

The builder records local media and subtitle paths in its output. Therefore
the Python `random` implementation, input normalization, JSON serialization,
newline convention, and especially the local path convention can affect the
manifest's byte-level SHA-256. The published hash is an accepted-run audit
target; matching selected IDs and order does not by itself guarantee the same
bytes. Exact hash matching requires the same normalized input and path
convention. Typical commands are in [`REPRODUCE.md`](REPRODUCE.md).

## HD-EPIC-51 cohort

The paper cohort contains 51 complete videos and every eligible question for
those videos: 1,871 question rows. A video contributes between 20 and 86
questions (minimum 20, median 33, mean 36.686, maximum 86). The shared
cumulative comparison horizon is the first 20 ordered questions; the
all-question endpoint uses all 1,871 rows.

Only official examples whose input is one complete video are eligible. Sort
eligible videos within participant, round-robin across participants, and retain
all questions for each selected video. Official question prototypes determine
the fixed within-video order. The 51-video participant distribution is:

| Participant | Videos | Participant | Videos | Participant | Videos |
| --- | ---: | --- | ---: | --- | ---: |
| P01 | 5 | P04 | 3 | P07 | 7 |
| P02 | 5 | P05 | 4 | P08 | 6 |
| P03 | 8 | P06 | 6 | P09 | 7 |

Audit identifiers:

- path-relative `questions.jsonl` SHA-256:
  `0c25ee80492909c026949c110519befce398ad06a6f770626e1424702efce3ae`
- locally materialized manifest SHA-256:
  `f4c19f5a828b98c34e3322882970bcb925cc0782834c112e1656aaf14427a2d6`
- official annotation snapshot SHA-256:
  `778d60c89c54fe771ef2bcc7e63cdc1d10b9364cbb6770ddbe88aefe6095b97c`

The audited annotation snapshot contains 26,550 questions, 100 fewer than the
count stated on the project page/paper. This is why the snapshot hash, rather
than only a dataset name, is part of the provenance record. The selected local
media totalled 70,533,652,710 bytes; that size is an integrity cross-check, not
a redistributable archive.

The path-relative manifest contents are not tracked, but the deterministic
builder is [`scripts/build_hdepic_mq10.py`](../scripts/build_hdepic_mq10.py).
Run it with 51 videos, a minimum of 20 questions, and
`--include-all-eligible-questions` to reproduce the selected IDs and ordering.
Then use
[`scripts/materialize_hdepic_mq10.py`](../scripts/materialize_hdepic_mq10.py)
to validate the downloaded videos and add local runtime paths. The
path-relative builder output should match the first hash above; the materialized
hash depends on the local absolute path and is retained as accepted-run
provenance.

## Frame and annotation handling

- The four main paths use the same 64 question-blind, uniformly spaced
  timestamps for a video. Timestamps are selected before any question is read.
- Frames are decoded from the original media; model-specific resizing and
  patch settings are listed in [`MODELS.md`](MODELS.md).
- CG subtitle text is never a model input.
- HD-EPIC `<TIME>` and `<BBOX>` references are normalized with the runner's
  `--expand-official-visual-references` option. The diagnostic option that adds
  a bounding-box reference frame is not part of the main four-path protocol.
- All paths attempt the same questions in the same order. Do not silently drop
  decode failures or malformed model answers.

Do not create a public smoke fixture by trimming a benchmark asset unless the
rights holder has explicitly permitted that redistribution.
