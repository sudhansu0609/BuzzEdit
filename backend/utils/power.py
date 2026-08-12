import ctypes
import logging
import sys

logger = logging.getLogger(__name__)

# Windows Execution State Flags
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002


class PowerManager:
    _lock_active = False

    @classmethod
    def prevent_sleep(cls, keep_display_on: bool = False):
        if sys.platform == "win32":
            try:
                flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED
                if keep_display_on:
                    flags |= ES_DISPLAY_REQUIRED
                
                result = ctypes.windll.kernel32.SetThreadExecutionState(flags)
                if result != 0:
                    cls._lock_active = True
                    logger.info("Windows sleep lock ACTIVATED (preventing system sleep during rendering)")
                else:
                    logger.warning("Failed to set Windows thread execution state")
            except Exception as e:
                logger.error(f"Error setting power execution state: {e}")
        else:
            logger.info("Non-Windows OS: Power sleep lock skipped")

    @classmethod
    def restore_sleep(cls):
        if sys.platform == "win32" and cls._lock_active:
            try:
                result = ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
                cls._lock_active = False
                logger.info("Windows sleep lock RELEASED (normal sleep restored)")
            except Exception as e:
                logger.error(f"Error releasing power execution state: {e}")
