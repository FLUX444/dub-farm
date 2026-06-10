from __future__ import annotations

import gc
import logging

logger = logging.getLogger(__name__)


def clear_gpu_memory() -> None:
    """Release PyTorch CUDA cache after a worker finishes."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            logger.debug("GPU memory cleared")
    except ImportError:
        pass
