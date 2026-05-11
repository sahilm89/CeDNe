"""Security tests for RestrictedUnpickler — verifies known-dangerous pandas
callables (eval/query and the pandas.core.computation subsystem) cannot be
resolved by find_class, even though "pandas" is an allowed module prefix."""

import io
import pickle

import pandas
import pytest

from cedne.core.io import RestrictedUnpickler


def _stream_resolving(module, name):
    """Build a minimal pickle stream that, on load, invokes find_class(module, name)
    and returns the resolved global. We construct it by hand-pickling a placeholder
    callable, then patching the GLOBAL opcode's module/name. Easier route: just
    call find_class directly on an Unpickler instance — that's what we do below."""
    return RestrictedUnpickler(io.BytesIO()).find_class(module, name)


def test_pandas_eval_top_level_is_denied():
    with pytest.raises(pickle.UnpicklingError, match="denied"):
        _stream_resolving("pandas", "eval")


def test_pandas_query_top_level_is_denied():
    with pytest.raises(pickle.UnpicklingError, match="denied"):
        _stream_resolving("pandas", "query")


def test_pandas_computation_submodule_is_denied():
    # pandas.eval's true __module__ — must also be blocked so the prefix bypass
    # via the canonical location is closed.
    assert pandas.eval.__module__.startswith("pandas.core.computation")
    with pytest.raises(pickle.UnpicklingError, match="denied"):
        _stream_resolving(pandas.eval.__module__, "eval")


def test_pandas_io_pickle_is_denied():
    with pytest.raises(pickle.UnpicklingError, match="denied"):
        _stream_resolving("pandas.io.pickle", "read_pickle")


def test_pandas_dataframe_still_allowed():
    # Make sure the deny list doesn't break legitimate pandas types that Worms
    # carry as attribute tables.
    cls = _stream_resolving("pandas.core.frame", "DataFrame")
    assert cls is pandas.DataFrame


def test_pandas_series_still_allowed():
    cls = _stream_resolving("pandas.core.series", "Series")
    assert cls is pandas.Series


def test_random_module_still_rejected():
    with pytest.raises(pickle.UnpicklingError, match="forbidden"):
        _stream_resolving("os", "system")


def test_end_to_end_crafted_pickle_rejected():
    """Build a real pickle stream that resolves pandas.eval, then try to load it
    via RestrictedUnpickler. Confirms the deny rule fires during a normal
    pickle.load path, not just direct find_class calls."""
    payload = pickle.dumps(pandas.eval)
    with pytest.raises(pickle.UnpicklingError, match="denied"):
        RestrictedUnpickler(io.BytesIO(payload)).load()
