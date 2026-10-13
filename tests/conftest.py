from pathlib import Path

import pandas as pd
import pytest

from src.data.oe import process

FIXTURE = Path(__file__).parent / "fixtures" / "oe_fixture.csv"


@pytest.fixture(scope="session")
def raw():
    return pd.read_csv(FIXTURE, low_memory=False)


@pytest.fixture(scope="session")
def processed(raw):
    return process(raw)
