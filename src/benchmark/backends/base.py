from __future__ import annotations

import abc
import logging
import subprocess
import time

import httpx

from benchmark.config import BackendConfig

logger = logging.getLogger(__name__)


class ServingBackend(abc.ABC):
    def __init__(self, config: BackendConfig) -> None:
        self.config = config
        self._process: subprocess.Popen | None = None

    @abc.abstractmethod
    def build_command(self) -> list[str]:
        ...

    def start(self, timeout: float = 300.0) -> None:
        cmd = self.build_command()
        logger.info("Starting %s: %s", self.config.backend_type.value, " ".join(cmd))

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self._wait_healthy(timeout)
        logger.info(
            "%s ready at %s", self.config.backend_type.value, self.config.base_url
        )

    def _wait_healthy(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._process and self._process.poll() is not None:
                raise RuntimeError(
                    f"Backend process exited with code {self._process.returncode}"
                )
            try:
                resp = httpx.get(self.config.health_url, timeout=2.0)
                if resp.status_code == 200:
                    return
            except httpx.ConnectError:
                pass
            time.sleep(2.0)
        raise TimeoutError(
            f"Backend did not become healthy within {timeout}s"
        )

    def stop(self) -> None:
        if self._process is None:
            return
        logger.info("Stopping %s", self.config.backend_type.value)
        self._process.terminate()
        try:
            self._process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=10)
        self._process = None

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def get_endpoint(self) -> str:
        return self.config.chat_url

    def __enter__(self) -> ServingBackend:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
