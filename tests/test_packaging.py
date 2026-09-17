"""Packaging invariants: compose, the Dockerfiles, and what they may claim.

These are cheap file assertions rather than builds, and they exist because
every defect they cover shipped once already and none of them was visible
from the Python test suite:

* a `worker` service that printed one line and exited, sitting in the
  dependency graph as though it served requests (ADR 0001);
* a frontend image whose container command was `npm run dev`, the
  development server, for a production deployment;
* `PROFILE` where the loader reads `SATQUERY_PROFILE`, so selecting the lite
  profile through compose silently did nothing.

A build takes minutes and needs a daemon; these run in milliseconds and catch
the regressions that actually happened.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ROOT / "docker-compose.yml"
API_DOCKERFILE = ROOT / "docker" / "api.Dockerfile"
WEB_DOCKERFILE = ROOT / "frontend" / "Dockerfile"


@pytest.fixture(scope="module")
def compose() -> dict:
    return yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))


class TestComposeTopology:
    def test_the_services_are_exactly_api_and_web(self, compose):
        """No third service, and specifically no worker. See ADR 0001."""
        assert set(compose["services"]) == {"api", "web"}

    def test_no_stub_worker_module_remains(self):
        """The module the removed service ran is gone, and nothing imports it."""
        assert not (ROOT / "satquery" / "controller" / "worker.py").exists()
        assert not (ROOT / "docker" / "worker.Dockerfile").exists()

        sources = [
            p for p in (ROOT / "satquery").rglob("*.py")
        ] + [p for p in (ROOT / "evaluation").rglob("*.py")]
        offenders = [
            p.relative_to(ROOT)
            for p in sources
            if "controller.worker" in p.read_text(encoding="utf-8")
        ]
        assert offenders == []

    def test_the_profile_variable_is_the_one_the_loader_reads(self, compose):
        from satquery.controller.profiles import ENV_PROFILE

        environment = compose["services"]["api"]["environment"]
        names = [entry.split("=", 1)[0] for entry in environment]
        assert ENV_PROFILE in names
        # The bug was a plausible-looking neighbour, so assert its absence too.
        assert "PROFILE" not in names

    def test_both_services_declare_a_healthcheck(self, compose):
        for name, service in compose["services"].items():
            assert "healthcheck" in service, f"{name} has no healthcheck"

    def test_the_web_service_waits_for_a_healthy_api(self, compose):
        """`depends_on` alone waits for the container, not for the app."""
        depends = compose["services"]["web"]["depends_on"]
        assert depends["api"]["condition"] == "service_healthy"

    def test_cors_origins_are_configurable_and_never_a_wildcard(self, compose):
        environment = compose["services"]["api"]["environment"]
        origins = next(e for e in environment if e.startswith("SATQUERY_CORS_ORIGINS="))
        assert "*" not in origins


class TestFrontendImage:
    def test_the_container_command_is_not_the_development_server(self):
        text = WEB_DOCKERFILE.read_text(encoding="utf-8")
        command = [
            line for line in text.splitlines() if line.startswith(("CMD", "ENTRYPOINT"))
        ]
        assert command, "no CMD in the frontend Dockerfile"
        assert all("dev" not in line for line in command), command

    def test_the_image_builds_the_application(self):
        text = WEB_DOCKERFILE.read_text(encoding="utf-8")
        assert "npm run build" in text
        # A lockfile-faithful install, not `npm install`, so the image gets
        # the dependency versions that were tested.
        assert "npm ci" in text

    def test_the_public_api_url_is_a_build_argument(self):
        """Next.js inlines NEXT_PUBLIC_* at build time; a runtime env var
        cannot reach an already-compiled bundle."""
        assert "ARG NEXT_PUBLIC_API_URL" in WEB_DOCKERFILE.read_text(encoding="utf-8")

    def test_the_compose_service_passes_it_as_a_build_argument(self, compose):
        args = compose["services"]["web"]["build"]["args"]
        assert any(str(a).startswith("NEXT_PUBLIC_API_URL=") for a in args)


class TestImagesRunUnprivileged:
    @pytest.mark.parametrize(
        "dockerfile", [API_DOCKERFILE, WEB_DOCKERFILE], ids=["api", "web"]
    )
    def test_a_non_root_user_is_selected(self, dockerfile):
        lines = [
            line.strip()
            for line in dockerfile.read_text(encoding="utf-8").splitlines()
            if line.startswith("USER ")
        ]
        assert lines, f"{dockerfile.name} never drops privileges"
        assert lines[-1] != "USER root"


class TestPythonBase:
    @pytest.mark.parametrize("dockerfile", [API_DOCKERFILE], ids=["api"])
    def test_the_base_image_can_resolve_the_pinned_requirements(self, dockerfile):
        """rasterio 1.5.1 publishes no wheels below Python 3.12.

        Both images were on 3.11 and could not build at all; the pin and the
        reason are recorded here so a future base bump cannot silently
        reintroduce it.
        """
        assert "python:3.12" in dockerfile.read_text(encoding="utf-8")
