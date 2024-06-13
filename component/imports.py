import os
import torch
import importlib
import importlib.metadata
from packaging import version

def _is_package_available(pkg_name, metadata_name=None):
    # Check we're not importing a "pkg_name" directory somewhere but the actual library by trying to grab the version
    package_exists = importlib.util.find_spec(pkg_name) is not None
    if package_exists:
        try:
            # Some libraries have different names in the metadata
            _ = importlib.metadata.metadata(pkg_name if metadata_name is None else metadata_name)
            return True
        except importlib.metadata.PackageNotFoundError:
            return False

def is_bnb_available():
    return _is_package_available("bitsandbytes")

def is_cuda_available():
    """
    Checks if `cuda` is available via an `nvml-based` check which won't trigger the drivers and leave cuda
    uninitialized.
    """
    pytorch_nvml_based_cuda_check_previous_value = os.environ.get("PYTORCH_NVML_BASED_CUDA_CHECK")
    try:
        os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = str(1)
        available = torch.cuda.is_available()
    finally:
        if pytorch_nvml_based_cuda_check_previous_value:
            os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = pytorch_nvml_based_cuda_check_previous_value
        else:
            os.environ.pop("PYTORCH_NVML_BASED_CUDA_CHECK", None)
    return available


def is_npu_available(check_device=False):
    "Checks if `torch_npu` is installed and potentially if a NPU is in the environment"
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("torch_npu") is None:
        return False

    import torch
    import torch_npu  # noqa: F401

    if check_device:
        try:
            # Will raise a RuntimeError if no NPU is found
            _ = torch.npu.device_count()
            return torch.npu.is_available()
        except RuntimeError:
            return False
    return hasattr(torch, "npu") and torch.npu.is_available()


