.PHONY: dev eval offline-test demo test soak adversarial calibrate selective entailment-bench lite-check report

dev:
	docker-compose up api web

eval:
	# Run the batch evaluation mode
	satquery eval --manifest data/manifest.jsonl --task vqa --out preds.jsonl

# Task 3.9. The HF_* variables stop the huggingface libraries reaching out;
# tests/test_offline.py additionally blocks the socket layer itself, because
# those variables say nothing about a stray requests.get or a library that
# quietly checks for updates.
offline-test:
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_PROXY=* python -m pytest tests/ -q

# The offline guarantee that actually matters: a cold boot with no network
# on the venue laptop. Runs only the socket-blocking suite, so a failure is
# unambiguous rather than buried in 600 other tests.
offline-check:
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest tests/test_offline.py -q

demo:
	# Run the demo bundle
	docker-compose up

test:
	python -m pytest tests/ -q

test-resume:
	python training/run_checkpoint_test.py

# --- Phase 3 evaluation targets --------------------------------------------

# Task 3.11. The plan asks for 20 queries; 20 is too short to separate a leak
# from warm-up (see docs/phase1-status.md), so the reported run is 120 with a
# 20-iteration warm-up excluded from the trend.
soak:
	python evaluation/soak.py --iterations 120 --warmup 20

# Task 3.8: 200 adversarial queries x 3 configurations.
adversarial:
	python evaluation/adversarial.py

# Task 3.3: fit per-head calibration and write the reliability diagrams.
calibrate:
	python evaluation/calibrate.py --heads landcover intent change_mask

# Task 3.6: risk-coverage curves and AURC from the cached logits.
selective:
	python evaluation/selective.py

# Task 3.5: score both entailment backends on both suites. Needs SATQUERY_NLI
# for --compare; without it only the deterministic backend is scored.
entailment-bench:
	python evaluation/entailment_bench.py $(if $(SATQUERY_NLI),--compare,)

# Task 3.10: every task must answer under the lite profile.
lite-check:
	python -m pytest tests/test_profiles.py -q

# Task 3.7: the four ablations.
ablations:
	python evaluation/run_ablations.py

report: calibrate selective adversarial soak entailment-bench ablations
	@echo "All Phase 3 evaluation artifacts regenerated under docs/assets/."
