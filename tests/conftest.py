import os
import tempfile

# The libraries pick their log directory on import: keep test runs out of ./flight_logs.
os.environ.setdefault("AERORIO_LOG_DIR", tempfile.mkdtemp(prefix="mission-delivery-tests-"))
