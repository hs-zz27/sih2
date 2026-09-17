"""CDVQA manifest preparation (plan task 2.6's evaluation half).

The imagery for CDVQA comes from a third-party mirror while the questions and
answers come from the official Apache-2.0 release. The whole safety of that
arrangement rests on one property: **a mirror sample whose question, answer or
question type disagrees with the official annotation is dropped, not used.**
These tests assert that property directly, because if it does not hold the
benchmark is silently corrupted and every CDVQA number we publish is wrong.

The fixtures are synthetic tars built in a tmp dir, so nothing here needs the
real 32 GB mirror or a network.
"""

from __future__ import annotations

import io
import json
import tarfile

import pytest

from training.prepare.cdvqa import build, build_from_second, load_official

# One official pair of image records, three questions, three answers.
OFFICIAL = {
    "Test_questions.json": {
        "questions": [
            {"id": 0, "img_id": 0, "type": "change_or_not",
             "question": "Did the areas of trees change?", "active": True},
            {"id": 1, "img_id": 0, "type": "change_ratio",
             "question": "How much of the area has changed?", "active": True},
            {"id": 2, "img_id": 1, "type": "change_or_not",
             "question": "Have the regions of water changed?", "active": True},
        ]
    },
    "Test_answers.json": {
        "answers": [
            {"id": 0, "question_id": 0, "answer": "yes", "active": True},
            {"id": 1, "question_id": 1, "answer": "10_to_20", "active": True},
            {"id": 2, "question_id": 2, "answer": "no", "active": True},
        ]
    },
    "Test_images.json": {
        "images": [
            {"id": 0, "file_name": "00031.png", "active": True},
            {"id": 1, "file_name": "00042.png", "active": True},
        ]
    },
}

PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc```\x00\x00"
    b"\x00\x04\x00\x01\xf6\x178U\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _sample(question_id, image_id, question, answer, question_type):
    return {
        "task": "change_detection",
        "num_images": 2,
        "conversations": [
            {"from": "user", "value": f"Image 1: <image>\nImage 2: <image>\n{question}"},
            {"from": "gpt", "value": answer},
        ],
        "meta": {
            "source": "CDVQA/SECOND",
            "split": "test",
            "image_id": image_id,
            "question_id": question_id,
            "question_type": question_type,
        },
    }


def _write_shard(path, samples):
    with tarfile.open(path, "w") as tar:
        for i, sample in enumerate(samples):
            key = f"cdvqa-test-{i:08d}"
            for member, blob in (
                (f"{key}.0.img", PNG),
                (f"{key}.1.img", PNG),
                (f"{key}.json", json.dumps(sample).encode()),
            ):
                info = tarfile.TarInfo(member)
                info.size = len(blob)
                tar.addfile(info, io.BytesIO(blob))


@pytest.fixture
def annotations(tmp_path):
    root = tmp_path / "cdvqa"
    root.mkdir()
    for name, body in OFFICIAL.items():
        (root / name).write_text(json.dumps(body), encoding="utf-8")
    return root


@pytest.fixture
def shards(tmp_path):
    d = tmp_path / "shards"
    d.mkdir()
    return d


def _build(annotations, shards, tmp_path, **kw):
    return build(
        annotations=annotations,
        shards=shards,
        split="Test",
        images_out=tmp_path / "out" / "images",
        manifest_out=tmp_path / "out" / "cdvqa_test.json",
        **kw,
    )


class TestOfficialAnnotations:
    def test_questions_answers_and_images_are_joined(self, annotations):
        official = load_official(annotations, "Test")
        assert official[0] == {
            "question": "Did the areas of trees change?",
            "answer": "yes",
            "type": "change_or_not",
            "image_id": "00031.png",
        }

    def test_a_missing_file_names_the_download_command(self, tmp_path):
        with pytest.raises(SystemExit, match="raw.githubusercontent.com"):
            load_official(tmp_path, "Test")


class TestMirrorVerification:
    def test_a_faithful_mirror_produces_every_item(self, annotations, shards, tmp_path):
        _write_shard(shards / "test-00000.tar", [
            _sample(0, "00031.png", "Did the areas of trees change?", "yes", "change_or_not"),
            _sample(1, "00031.png", "How much of the area has changed?", "10_to_20", "change_ratio"),
        ])
        report = _build(annotations, shards, tmp_path)
        assert report["n_items"] == 2
        assert report["dropped_mismatched"] == 0

    @pytest.mark.parametrize(
        "sample",
        [
            # Drifted question text for the right question id.
            _sample(0, "00031.png", "Did the areas of water change?", "yes", "change_or_not"),
            # Drifted answer.
            _sample(0, "00031.png", "Did the areas of trees change?", "no", "change_or_not"),
            # Drifted question type.
            _sample(0, "00031.png", "Did the areas of trees change?", "yes", "change_ratio"),
        ],
        ids=["question", "answer", "type"],
    )
    def test_a_drifted_sample_is_dropped_not_used(
        self, annotations, shards, tmp_path, sample
    ):
        """A mirror that disagrees with the official release shrinks the
        manifest. Silently taking its word would corrupt the benchmark."""
        _write_shard(shards / "test-00000.tar", [sample])
        report = _build(annotations, shards, tmp_path)
        assert report["n_items"] == 0
        assert report["dropped_mismatched"] == 1

    def test_a_question_id_outside_the_split_is_dropped(
        self, annotations, shards, tmp_path
    ):
        _write_shard(shards / "test-00000.tar", [
            _sample(999, "00031.png", "Did the areas of trees change?", "yes", "change_or_not"),
        ])
        report = _build(annotations, shards, tmp_path)
        assert report["n_items"] == 0
        assert report["dropped_unknown_question_id"] == 1


class TestManifest:
    def test_image_pairs_are_written_once_and_shared(
        self, annotations, shards, tmp_path
    ):
        """The mirror stores a copy of the pair per question; writing every
        copy is what makes the full split 32 GB for 968 pairs."""
        _write_shard(shards / "test-00000.tar", [
            _sample(0, "00031.png", "Did the areas of trees change?", "yes", "change_or_not"),
            _sample(1, "00031.png", "How much of the area has changed?", "10_to_20", "change_ratio"),
            _sample(2, "00042.png", "Have the regions of water changed?", "no", "change_or_not"),
        ])
        report = _build(annotations, shards, tmp_path)

        assert report["n_items"] == 3
        assert report["n_pairs"] == 2
        assert sorted(p.name for p in (tmp_path / "out" / "images").iterdir()) == [
            "00031_t1.png", "00031_t2.png", "00042_t1.png", "00042_t2.png",
        ]

    def test_manifest_matches_the_harness_contract(self, annotations, shards, tmp_path):
        _write_shard(shards / "test-00000.tar", [
            _sample(0, "00031.png", "Did the areas of trees change?", "yes", "change_or_not"),
        ])
        _build(annotations, shards, tmp_path)

        from evaluation.harness import load_benchmark

        items = load_benchmark(tmp_path / "out" / "cdvqa_test.json")
        assert items == [
            {
                "item_id": "cdvqa_test_000000",
                "images": ["images/00031_t1.png", "images/00031_t2.png"],
                "question": "Did the areas of trees change?",
                "answer": "yes",
                # The harness breaks accuracy down by answer_type, which is how
                # CDVQA's "accuracy per question type" gets reported for free.
                "answer_type": "change_or_not",
            }
        ]

    def test_duplicate_question_ids_across_shards_appear_once(
        self, annotations, shards, tmp_path
    ):
        sample = _sample(0, "00031.png", "Did the areas of trees change?", "yes", "change_or_not")
        _write_shard(shards / "test-00000.tar", [sample])
        _write_shard(shards / "test-00001.tar", [sample])
        report = _build(annotations, shards, tmp_path)
        assert report["n_items"] == 1

    def test_a_partial_shard_is_skipped_rather_than_fatal(
        self, annotations, shards, tmp_path
    ):
        """Shards arrive by parallel download; a half-written one must cost
        coverage, not the whole run."""
        _write_shard(shards / "test-00000.tar", [
            _sample(0, "00031.png", "Did the areas of trees change?", "yes", "change_or_not"),
        ])
        (shards / "test-00001.tar").write_bytes(b"not a tar, still downloading")

        report = _build(annotations, shards, tmp_path)
        assert report["n_items"] == 1
        assert report["shards_unreadable"] == ["test-00001.tar"]

    def test_coverage_is_reported_against_the_official_split(
        self, annotations, shards, tmp_path
    ):
        """A partial download must announce how partial it is, or the accuracy
        it produces reads as a full-split number."""
        _write_shard(shards / "test-00000.tar", [
            _sample(0, "00031.png", "Did the areas of trees change?", "yes", "change_or_not"),
        ])
        report = _build(annotations, shards, tmp_path)
        assert report["official_questions"] == 3
        assert report["official_pairs"] == 2
        assert report["question_coverage"] == pytest.approx(1 / 3)

    def test_no_shards_names_the_download_command(self, annotations, shards, tmp_path):
        with pytest.raises(SystemExit, match="huggingface.co"):
            _build(annotations, shards, tmp_path)

    def test_limit_caps_the_manifest(self, annotations, shards, tmp_path):
        _write_shard(shards / "test-00000.tar", [
            _sample(0, "00031.png", "Did the areas of trees change?", "yes", "change_or_not"),
            _sample(1, "00031.png", "How much of the area has changed?", "10_to_20", "change_ratio"),
        ])
        report = _build(annotations, shards, tmp_path, limit=1)
        assert report["n_items"] == 1


class TestSecondSource:
    """SECOND is the better image source, and the reason is coverage.

    CDVQA's splits partition SECOND's 2,968 labelled pairs, so every image id
    resolves and the manifest reaches 100% instead of whatever fraction of the
    32 GB mirror finished downloading.
    """

    @pytest.fixture
    def second(self, tmp_path):
        root = tmp_path / "second"
        for sub in ("im1", "im2"):
            (root / sub).mkdir(parents=True)
            for name in ("00031.png", "00042.png"):
                (root / sub / name).write_bytes(PNG)
        return root

    def test_every_official_question_appears(self, annotations, second, tmp_path):
        out = tmp_path / "out" / "cdvqa_test.json"
        report = build_from_second(annotations, second, "Test", out)

        assert report["n_items"] == report["official_questions"] == 3
        assert report["question_coverage"] == 1.0
        assert report["n_pairs"] == 2

    def test_the_manifest_points_at_second_and_loads(
        self, annotations, second, tmp_path
    ):
        from evaluation.harness import load_benchmark

        out = tmp_path / "out" / "cdvqa_test.json"
        build_from_second(annotations, second, "Test", out)
        items = load_benchmark(out)

        assert len(items) == 3
        assert items[0]["item_id"] == "cdvqa_test_000000"
        assert items[0]["answer_type"] == "change_or_not"
        for image in items[0]["images"]:
            assert (out.parent / image).exists()

    def test_an_id_missing_from_second_is_counted_not_fatal(
        self, annotations, second, tmp_path
    ):
        (second / "im1" / "00042.png").unlink()
        report = build_from_second(
            annotations, second, "Test", tmp_path / "out" / "m.json"
        )
        assert report["n_items"] == 2
        assert report["dropped_incomplete_sample"] == 1

    def test_a_missing_second_root_names_the_download(self, annotations, tmp_path):
        with pytest.raises(SystemExit, match="SECOND_train_set"):
            build_from_second(
                annotations, tmp_path / "absent", "Test", tmp_path / "m.json"
            )
