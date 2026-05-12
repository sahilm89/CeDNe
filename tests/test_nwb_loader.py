import os
import numpy as np
import pytest
from datetime import datetime
from pynwb import NWBFile, NWBHDF5IO
from pynwb.file import Subject
from cedne.utils.loader import load_nwb
from cedne.core import Worm, NervousSystem, Session


def create_mock_nwb(filename):
    """Creates a mock NWB file for testing."""
    nwbfile = NWBFile(
        session_description="test session",
        identifier="test_id",
        session_start_time=datetime.now().astimezone(),
        experimenter="Sydney Brenner",
        lab="Brenner Lab",
        institution="Caltech",
        session_id="test_session_001",
    )

    nwbfile.subject = Subject(
        species="C. elegans", strain="N2", age="Adult", sex="Hermaphrodite"
    )

    # Add Units (Spikes)
    nwbfile.add_unit_column("label", "a label for the unit")
    nwbfile.add_unit(spike_times=[1.0, 2.0, 3.0], label="Neuron_A")
    nwbfile.add_unit(spike_times=[0.5, 1.5], label="Neuron_B")

    with NWBHDF5IO(filename, "w") as io:
        io.write(nwbfile)


def test_load_nwb_basic(tmp_path):
    nwb_path = os.path.join(tmp_path, "test.nwb")
    create_mock_nwb(nwb_path)

    nn, session = load_nwb(nwb_path)

    # Verify NervousSystem and Worm
    assert isinstance(nn, NervousSystem)
    assert isinstance(nn.worm, Worm)
    assert nn.worm.name == "test_session_001"

    # Verify Session
    assert isinstance(session, Session)
    assert session.experimenter == ("Sydney Brenner",)  # pynwb might return a tuple
    assert session.lab == "Brenner Lab"

    # Verify Neurons and Trials
    assert "Unit_0" in nn.neurons
    assert "Unit_1" in nn.neurons

    neuron_a = nn.neurons["Unit_0"]
    assert 0 in neuron_a.trial
    assert np.array_equal(neuron_a.trial[0].recording, [1.0, 2.0, 3.0])

    # Verify Session Trial mapping
    assert 0 in session.trials
    assert len(session.trials[0]) == 2  # Two neurons added trials


def test_load_nwb_no_file():
    with pytest.raises(FileNotFoundError):
        load_nwb("non_existent_file.nwb")


if __name__ == "__main__":
    # For manual testing
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        test_load_nwb_basic(tmpdir)
        print("Test basic passed!")
