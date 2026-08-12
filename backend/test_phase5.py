import os
import sys
import logging
from pathlib import Path

backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from utils.gpu_utils import get_gpu_info, clear_vram_cache, is_oom_error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_phase5")

def run_test():
    print("--- STARTING PHASE 5 PRODUCTION HARDENING TEST ---")
    
    # 1. Test GPU Info Query
    print("[1/3] Querying GPU VRAM Status...")
    info = get_gpu_info()
    print(f"GPU Info: {info}")
    assert "available" in info
    assert "device_name" in info
    
    # 2. Test VRAM Cache Clearing
    print("[2/3] Testing PyTorch VRAM Cache Clearing...")
    clear_vram_cache()
    print("Cache clearing executed without error.")
    
    # 3. Test OOM Exception Inspector
    print("[3/3] Testing OOM Exception Detection...")
    oom_exc = RuntimeError("CUDA error: out of memory. Tried to allocate 2.00 GiB")
    normal_exc = ValueError("Invalid parameter value")
    
    assert is_oom_error(oom_exc) == True, "Should recognize CUDA OOM exception"
    assert is_oom_error(normal_exc) == False, "Should not match regular exception"
    
    print("[4/4] PHASE 5 VERIFICATION SUCCESSFUL!")

if __name__ == "__main__":
    run_test()
