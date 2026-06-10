from pathlib import Path
import shutil

import pytest


@pytest.fixture(autouse=True)
def clean_prereg_dir():
    prereg = Path("prereg")
    shutil.rmtree(prereg, ignore_errors=True)
    yield
    shutil.rmtree(prereg, ignore_errors=True)
