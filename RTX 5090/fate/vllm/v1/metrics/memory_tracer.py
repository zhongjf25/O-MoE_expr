import csv
import time
from typing import Optional

from vllm.logger import init_logger

logger = init_logger(__name__)

GiB = 1 << 30


class MemoryTracer:
    """Traces per-step GPU memory breakdown (dense / expert / KV / activation)
    to a CSV file.  Controlled by the VLLM_MEMORY_TRACE environment variable.
    """

    def __init__(
        self,
        output_path: str,
        huge_page_size: int,
        dense_weight_bytes: int,
    ) -> None:
        self._file = open(output_path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow([
            "time_s", "step",
            "dense_gb", "expert_gb", "kv_gb", "act_gb",
        ])
        self._start = time.monotonic()
        self._hps = huge_page_size
        self._dense = dense_weight_bytes
        self._step = 0
        logger.info("Memory tracer enabled, writing to %s", output_path)

    def record(self, expert_blocks: int, kv_blocks: int,
               act_blocks: int) -> None:
        t = time.monotonic() - self._start
        hps = self._hps
        self._writer.writerow([
            f"{t:.3f}",
            self._step,
            f"{self._dense / GiB:.4f}",
            f"{expert_blocks * hps / GiB:.4f}",
            f"{kv_blocks * hps / GiB:.4f}",
            f"{act_blocks * hps / GiB:.4f}",
        ])
        self._step += 1
        if self._step % 50 == 0:
            self._file.flush()

    def close(self) -> None:
        self._file.flush()
        self._file.close()
        logger.info("Memory tracer closed after %d steps", self._step)
