from typing import Literal

GpuT = Literal["geforce3090", "v100", "a6000", "a40", "l40", "a100", "h100"]


def ampere_or_newer_gpu(gpu_type: GpuT) -> bool:
    match gpu_type:
        case "geforce3090" | "a6000" | "a40" | "l40" | "a100" | "h100":
            return True
        case "v100":
            return False
