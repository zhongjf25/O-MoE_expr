
import threading

# A tiny helper to expose a thread-local profiling flag that downstream
# code (attention backends, MoE layers, expert managers) can query to
# decide whether to perform real GPU allocations / prefetches during
# profile_run.

_tls = threading.local()

def set_profile_mode(v: bool):
    """Set whether we're running in profile mode on this thread."""
    setattr(_tls, "_is_profile", bool(v))

def get_profile_mode() -> bool:
    """Return True if this thread is in profile mode.

    Default is False.
    """
    return getattr(_tls, "_is_profile", False)
