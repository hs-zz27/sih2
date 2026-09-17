"""Phase 5 campaign harness: running the full training campaign on a shared
GPU cluster reached only through a notebook.

Every module here is importable without torch, because the campaign
configuration, the run queue and the staging manifest are all pure path and
JSON logic and are unit tested in CI, which does not install torch.
"""
