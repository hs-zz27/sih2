from abc import ABC, abstractmethod
from typing import Any

from satquery.contracts.input_manifest import InputManifest
from satquery.contracts.tool_result import ToolResult


class ToolProtocol(ABC):
    """
    Interface that all SatQuery tools (real or stub) must implement.
    """

    @abstractmethod
    def run(self, manifest: InputManifest, params: dict[str, Any]) -> ToolResult:
        """
        Execute the tool on a single input manifest.
        """
        pass

    @abstractmethod
    def run_batch(self, manifests: list[InputManifest], params: dict[str, Any]) -> list[ToolResult]:
        """
        Execute the tool on a batch of input manifests.
        """
        pass
