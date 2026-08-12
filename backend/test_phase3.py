import os
import sys
import logging
from pathlib import Path

backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from utils.power import PowerManager
from agents.scheduler import JobScheduler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_phase3")

def run_test():
    print("--- STARTING PHASE 3 NIGHTLY AUTOMATION TEST ---")
    
    # 1. Test Power Management
    print("[1/4] Testing Windows Power Sleep Lock...")
    PowerManager.prevent_sleep(keep_display_on=False)
    assert PowerManager._lock_active == (sys.platform == "win32")
    PowerManager.restore_sleep()
    assert not PowerManager._lock_active
    print("Power lock test passed.")
    
    # 2. Test Scheduler Job Enqueue & Priority Sorting
    print("[2/4] Testing Job Queue & Priority Sorting...")
    scheduler = JobScheduler()
    scheduler.jobs = [] # reset
    
    job_low = scheduler.enqueue_job("proj_low", priority="low")
    job_urgent = scheduler.enqueue_job("proj_urgent", priority="urgent")
    job_norm = scheduler.enqueue_job("proj_normal", priority="normal")
    
    assert scheduler.jobs[0]["id"] == job_urgent["id"], "Urgent job should be first"
    assert scheduler.jobs[1]["id"] == job_norm["id"], "Normal job should be second"
    assert scheduler.jobs[2]["id"] == job_low["id"], "Low priority job should be last"
    print("Priority sorting passed.")
    
    # 3. Test Job Cancellation & Clearing
    print("[3/4] Testing Job Cancellation & Clear Completed...")
    scheduler.cancel_job(job_low["id"])
    assert scheduler.jobs[2]["status"] == "cancelled"
    
    scheduler.clear_completed()
    assert len(scheduler.jobs) == 2, "Cancelled job should be cleared"
    print("Job cancellation passed.")
    
    print("[4/4] PHASE 3 VERIFICATION SUCCESSFUL!")

if __name__ == "__main__":
    run_test()
