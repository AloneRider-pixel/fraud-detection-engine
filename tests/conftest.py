import os
import subprocess
import sys

import pytest


@pytest.fixture(scope="session")
def artifacts(tmp_path_factory):
    d = tmp_path_factory.mktemp("art")
    subprocess.run([sys.executable, "-m", "fraud_engine.train", "--rows", "8000", "--out", str(d)],
                   check=True, capture_output=True)
    return str(d)
