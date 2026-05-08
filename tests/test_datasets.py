"""Tests for the dataset registry and MissingDatasetError surface.

Loader hardening (Issue 5 in CeDNe_web): every external read goes
through ``require_dataset_file``, and the resulting failure carries a
structured ``dataset_key`` so the web backend can render an actionable
banner instead of a raw ``[Errno 2]``.
"""
import hashlib
import io
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from cedne.utils.datasets import (
    DATASET_REGISTRY,
    DatasetSpec,
    DownloadSpec,
    DownloadResult,
    MissingDatasetError,
    download_all_public,
    download_dataset,
    require_dataset_file,
)
from cedne.utils import datasets as datasets_mod
from cedne.utils import loader


def test_registry_keys_are_snake_case_and_match_dataclass_field():
    """Every entry's dict key must equal the spec's `key` attribute,
    so callers can round-trip via ``DATASET_REGISTRY[spec.key]``.
    """
    for key, spec in DATASET_REGISTRY.items():
        assert isinstance(spec, DatasetSpec)
        assert spec.key == key
        assert key.islower()


def test_registry_covers_loader_dataset_keys():
    """The registry is the canonical source — every key referenced in
    ``loader.py`` must be defined here so the backend / UI can look up
    metadata for any error that escapes a loader.
    """
    expected_keys = {
        "cengen", "cook_2019", "witvliet_2020", "white_1986", "fly_wire",
        "winding_2023", "ryan_2016", "bumbarger_2013", "veraszto_2025",
        "brittin_2018", "ripoll_sanchez_2023", "randi_2023", "wang_2024",
        "atanas_2023", "worm_atlas_lineage",
    }
    assert expected_keys.issubset(DATASET_REGISTRY.keys())


def test_require_dataset_file_returns_path_when_present(tmp_path):
    real = tmp_path / "ok.txt"
    real.write_text("present")
    out = require_dataset_file(real, "cook_2019")
    assert out == real
    assert isinstance(out, Path)


def test_require_dataset_file_raises_missing_dataset_error(tmp_path):
    missing = tmp_path / "ghost.csv"
    with pytest.raises(MissingDatasetError) as exc_info:
        require_dataset_file(missing, "cook_2019")
    err = exc_info.value
    assert err.dataset_key == "cook_2019"
    assert err.expected_path == missing
    # Inherits from FileNotFoundError so existing optional-load fallbacks
    # in the loader (e.g. _readLigandTable) keep working.
    assert isinstance(err, FileNotFoundError)


def test_missing_dataset_error_payload_includes_metadata():
    err = MissingDatasetError(
        dataset_key="cook_2019",
        expected_path="/tmp/missing.xlsx",
        hint="stage on the data PVC",
    )
    payload = err.to_payload()
    assert payload["error"] == "missing_dataset"
    assert payload["dataset_key"] == "cook_2019"
    assert payload["expected_path"] == "/tmp/missing.xlsx"
    assert "Cook" in payload["title"]
    assert payload["citation"]
    assert payload["source_url"]
    assert payload["hint"] == "stage on the data PVC"
    msg = str(err)
    assert "cook_2019" in msg
    assert "/tmp/missing.xlsx" in msg
    assert "stage on the data PVC" in msg


def test_missing_dataset_error_unknown_key_falls_back_gracefully():
    """An unregistered key still produces a useful error — the formatter
    must not assume registry coverage.
    """
    err = MissingDatasetError(
        dataset_key="not_in_registry",
        expected_path="/tmp/x",
    )
    payload = err.to_payload()
    assert payload["dataset_key"] == "not_in_registry"
    assert payload["title"] == "not_in_registry"
    assert payload["citation"] is None
    assert payload["source_url"] is None


def test_loader_raises_missing_dataset_error_with_correct_key(tmp_path, monkeypatch):
    """Smoke test the wrapping for a representative loader: redirect the
    Cook connectome path to an empty tmpdir, expect a MissingDatasetError
    with key='cook_2019' rather than a raw FileNotFoundError.
    """
    monkeypatch.setattr(loader, "cook_connectome", tmp_path / "empty_cook")

    with pytest.raises(MissingDatasetError) as exc_info:
        loader.makeWorm(
            name="test",
            import_parameters={
                "style": "cook",
                "sex": "male",
                "stage": "adult",
                "dataset_ind": 1,
            },
        )
    assert exc_info.value.dataset_key == "cook_2019"


def test_loader_raises_missing_dataset_error_for_winding(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "winding_connectome", tmp_path / "empty_winding")

    with pytest.raises(MissingDatasetError) as exc_info:
        loader.makeFly(
            name="larva",
            import_parameters={"style": "Winding_2023"},
        )
    assert exc_info.value.dataset_key == "winding_2023"


def test_optional_ligand_table_fallback_still_works(tmp_path, monkeypatch):
    """``loadNeurotransmitters`` does ``except FileNotFoundError`` around
    the optional ligand-table read so the loader degrades gracefully when
    the table is absent. Because ``MissingDatasetError`` inherits from
    ``FileNotFoundError``, that contract must still hold.
    """
    monkeypatch.setattr(loader, "DOWNLOAD_DIR", tmp_path / "empty_nt")
    with pytest.raises(MissingDatasetError):
        # _readLigandTable raises directly when the file is gone; the
        # caller (loadNeurotransmitters) catches FileNotFoundError, but
        # exercising the helper alone is enough to prove inheritance.
        loader._readLigandTable(sex="Hermaphrodite")


# ---------------------------------------------------------------------------
# Step-2 infrastructure tests: DownloadSpec / download_dataset / archive
# extraction / sha256 verification / skip-when-present / all_public group.
# ---------------------------------------------------------------------------


def _fake_response(payload: bytes):
    """Build a stand-in for `requests.get(stream=True)` that yields
    `payload` in one chunk. Supports the context-manager + iter_content
    interface the real download helper uses.
    """
    resp = MagicMock()
    resp.__enter__ = lambda self_: self_
    resp.__exit__ = lambda *_: False
    resp.iter_content = lambda chunk_size=None: iter([payload])
    resp.raise_for_status = lambda: None
    return resp


def _patch_requests_get(monkeypatch, payload_for_url):
    """Monkeypatch the local `requests.get` that `_stream_download`
    imports lazily. `payload_for_url` is a dict[url -> bytes] OR a callable.
    """
    fake_requests = MagicMock()

    def fake_get(url, stream=False, timeout=None):
        if callable(payload_for_url):
            data = payload_for_url(url)
        else:
            data = payload_for_url[url]
        return _fake_response(data)

    fake_requests.get = fake_get
    monkeypatch.setitem(__import__("sys").modules, "requests", fake_requests)


def test_download_dataset_writes_files_and_records_status(tmp_path, monkeypatch):
    """Happy path: a one-spec dataset downloads the file, returns a
    DownloadResult with status='downloaded' and the actual sha256, and
    materialises the file at the registered target path.
    """
    payload = b"hello,world\n"
    expected_sha = hashlib.sha256(payload).hexdigest()
    spec = DatasetSpec(
        key="test_ok",
        title="Test dataset",
        expected_dir=tmp_path / "test_ok",
        download_specs=(DownloadSpec(
            url="https://example.com/data.csv",
            target_relpath="data.csv",
            sha256=expected_sha,
        ),),
    )
    monkeypatch.setitem(DATASET_REGISTRY, "test_ok", spec)
    _patch_requests_get(monkeypatch, {spec.download_specs[0].url: payload})

    results = download_dataset("test_ok", progress=lambda _msg: None)

    assert len(results) == 1
    r = results[0]
    assert r.status == "downloaded"
    assert r.sha256 == expected_sha
    assert r.target.read_bytes() == payload


def test_download_dataset_rejects_sha256_mismatch(tmp_path, monkeypatch):
    """If the downloaded bytes don't match the registered sha256, the
    file must be removed and the result must report 'failed' with the
    error message — never leave a corrupt file in place.
    """
    payload = b"actual content"
    wrong_sha = "0" * 64
    spec = DatasetSpec(
        key="test_bad_sha",
        title="Test dataset",
        expected_dir=tmp_path / "bad",
        download_specs=(DownloadSpec(
            url="https://example.com/x.csv",
            target_relpath="x.csv",
            sha256=wrong_sha,
        ),),
    )
    monkeypatch.setitem(DATASET_REGISTRY, "test_bad_sha", spec)
    _patch_requests_get(monkeypatch, {spec.download_specs[0].url: payload})

    results = download_dataset("test_bad_sha", progress=lambda _: None)

    assert results[0].status == "failed"
    assert "sha256 mismatch" in results[0].error
    assert not (spec.expected_dir / "x.csv").exists()


def test_download_dataset_skips_when_sha256_already_matches(tmp_path, monkeypatch):
    """A second invocation must short-circuit when the file is already
    on disk and its sha256 matches — no HTTP call.
    """
    payload = b"already here"
    sha = hashlib.sha256(payload).hexdigest()
    target_dir = tmp_path / "skip"
    target_dir.mkdir()
    (target_dir / "x.csv").write_bytes(payload)

    spec = DatasetSpec(
        key="test_skip",
        title="Test dataset",
        expected_dir=target_dir,
        download_specs=(DownloadSpec(
            url="https://example.com/x.csv",
            target_relpath="x.csv",
            sha256=sha,
        ),),
    )
    monkeypatch.setitem(DATASET_REGISTRY, "test_skip", spec)

    # If `requests.get` is hit, fail loudly — we expect zero network calls.
    def boom(*_a, **_k):
        raise AssertionError("requests.get should not be called for skip path")
    fake_requests = MagicMock()
    fake_requests.get = boom
    monkeypatch.setitem(__import__("sys").modules, "requests", fake_requests)

    results = download_dataset("test_skip", progress=lambda _: None)
    assert results[0].status == "skipped"
    assert results[0].sha256 == sha


def test_download_dataset_force_redownloads(tmp_path, monkeypatch):
    """`force=True` must re-download even when the file exists with a
    matching sha256."""
    payload = b"force me"
    sha = hashlib.sha256(payload).hexdigest()
    target_dir = tmp_path / "force"
    target_dir.mkdir()
    (target_dir / "f.csv").write_bytes(payload)

    spec = DatasetSpec(
        key="test_force",
        title="Test dataset",
        expected_dir=target_dir,
        download_specs=(DownloadSpec(
            url="https://example.com/f.csv",
            target_relpath="f.csv",
            sha256=sha,
        ),),
    )
    monkeypatch.setitem(DATASET_REGISTRY, "test_force", spec)
    calls = {"n": 0}

    def fake_get(url, **_k):
        calls["n"] += 1
        return _fake_response(payload)
    fake_requests = MagicMock(); fake_requests.get = fake_get
    monkeypatch.setitem(__import__("sys").modules, "requests", fake_requests)

    download_dataset("test_force", force=True, progress=lambda _: None)
    assert calls["n"] == 1


def test_download_dataset_extracts_zip_with_archive_keep_under(tmp_path, monkeypatch):
    """The Ripoll-Sanchez case: a GitHub repo archive bundles many top-level
    directories, but the loader only reads one. ``strip_prefix`` drops the
    repo-name wrapper; ``archive_keep_under`` filters extraction to a single
    subtree and removes that prefix from the placed paths.
    """
    # Mirror the GitHub archive shape: one outer wrapper, two siblings
    # under it, only one of which we want.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("repo-sha/Adjacency matrices for networks/pairs.csv", "id,name\n")
        z.writestr("repo-sha/Adjacency matrices for networks/sub/net001.csv", "row\n")
        z.writestr("repo-sha/Scripts & data/ignore.R", "x <- 1\n")
        z.writestr("repo-sha/README.md", "ignored\n")
    archive_bytes = buf.getvalue()
    sha = hashlib.sha256(archive_bytes).hexdigest()

    spec = DatasetSpec(
        key="test_keep_under",
        title="t",
        expected_dir=tmp_path / "kept",
        download_specs=(DownloadSpec(
            url="https://example.com/repo.zip",
            target_relpath="_archive.zip",
            sha256=sha,
            extract_to="new",
            strip_prefix=1,
            archive_keep_under="Adjacency matrices for networks/",
        ),),
    )
    monkeypatch.setitem(DATASET_REGISTRY, "test_keep_under", spec)
    _patch_requests_get(monkeypatch, {spec.download_specs[0].url: archive_bytes})

    download_dataset("test_keep_under", progress=lambda _: None)

    base = spec.expected_dir / "new"
    assert (base / "pairs.csv").exists()
    assert (base / "sub" / "net001.csv").exists()
    # Sibling directories must not have been extracted.
    assert not (base / "Scripts & data").exists()
    assert not (base / "README.md").exists()
    assert not (spec.expected_dir / "_archive.zip").exists()


def test_download_dataset_extracts_zip_with_strip_prefix(tmp_path, monkeypatch):
    """A zip archive with a single top-level directory must be extracted
    flat into expected_dir when strip_prefix=1 is set, mirroring the
    Winding 2023 case (Supplementary-Data-S1/<files> → expected_dir/<files>).
    """
    # Build an in-memory zip whose entries are namespaced under "wrap/".
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("wrap/a.csv", "row1\n")
        z.writestr("wrap/sub/b.csv", "row2\n")
        z.writestr("__MACOSX/wrap/._a.csv", "junk")  # must be filtered
    archive_bytes = buf.getvalue()
    sha = hashlib.sha256(archive_bytes).hexdigest()

    spec = DatasetSpec(
        key="test_zip",
        title="Test zip dataset",
        expected_dir=tmp_path / "zipdest",
        download_specs=(DownloadSpec(
            url="https://example.com/data.zip",
            target_relpath="data.zip",
            sha256=sha,
            extract_to="",
            strip_prefix=1,
        ),),
    )
    monkeypatch.setitem(DATASET_REGISTRY, "test_zip", spec)
    _patch_requests_get(monkeypatch, {spec.download_specs[0].url: archive_bytes})

    results = download_dataset("test_zip", progress=lambda _: None)

    assert results[0].status == "extracted"
    assert (spec.expected_dir / "a.csv").exists()
    assert (spec.expected_dir / "sub" / "b.csv").exists()
    # __MACOSX entries skipped, archive cleaned up by default.
    assert not (spec.expected_dir / "__MACOSX").exists()
    assert not (spec.expected_dir / "data.zip").exists()


def test_download_dataset_unknown_key_raises_keyerror():
    with pytest.raises(KeyError):
        download_dataset("not_a_real_key")


def test_download_dataset_no_specs_raises_valueerror(tmp_path, monkeypatch):
    """A registered dataset with empty download_specs must refuse rather
    than silently no-op — the user explicitly asked for a download."""
    spec = DatasetSpec(
        key="test_no_specs",
        title="t",
        expected_dir=tmp_path / "x",
    )
    monkeypatch.setitem(DATASET_REGISTRY, "test_no_specs", spec)
    with pytest.raises(ValueError, match="no registered download URLs"):
        download_dataset("test_no_specs")


def test_download_all_public_iterates_only_specs_with_urls(tmp_path, monkeypatch):
    """`all_public` must visit every registry entry that has at least one
    download_spec, and skip those without — reporting both clearly so
    the caller can tell the difference."""
    # Replace the registry entirely so the test's behaviour is independent
    # of how many real datasets happen to be registered.
    payload = b"abc"
    sha = hashlib.sha256(payload).hexdigest()
    fake_registry = {
        "with_specs": DatasetSpec(
            key="with_specs", title="t", expected_dir=tmp_path / "a",
            download_specs=(DownloadSpec(
                url="https://example.com/a.csv",
                target_relpath="a.csv", sha256=sha,
            ),),
        ),
        "no_specs": DatasetSpec(
            key="no_specs", title="t", expected_dir=tmp_path / "b",
            source_url="https://example.com/manual",
        ),
    }
    monkeypatch.setattr(datasets_mod, "DATASET_REGISTRY", fake_registry)
    _patch_requests_get(monkeypatch, {"https://example.com/a.csv": payload})

    summary = datasets_mod.download_all_public(progress=lambda _: None)

    assert set(summary.keys()) == {"with_specs", "no_specs"}
    assert len(summary["with_specs"]) == 1
    assert summary["with_specs"][0].status == "downloaded"
    assert summary["no_specs"] == []  # signaled as "no URLs registered"


def test_download_datasets_legacy_alias(tmp_path, monkeypatch):
    """The legacy ``download_datasets('atanas_whole_brain')`` key must
    still work — aliased to the canonical 'atanas_2023'."""
    # Stub out atanas_2023 to a tiny one-spec entry so we don't actually
    # try to fetch 39 files.
    payload = b"json:{}"
    sha = hashlib.sha256(payload).hexdigest()
    monkeypatch.setitem(
        DATASET_REGISTRY, "atanas_2023",
        DatasetSpec(
            key="atanas_2023", title="t", expected_dir=tmp_path / "atanas",
            download_specs=(DownloadSpec(
                url="https://example.com/x.json",
                target_relpath="Control/x.json", sha256=sha,
            ),),
        ),
    )
    _patch_requests_get(monkeypatch, {"https://example.com/x.json": payload})

    results = loader.download_datasets("atanas_whole_brain")
    assert len(results) == 1
    assert results[0].status == "downloaded"
    assert (tmp_path / "atanas" / "Control" / "x.json").exists()


def test_ripoll_sanchez_archive_extracts_to_layout_loader_expects(tmp_path, monkeypatch):
    """End-to-end check: the registered Ripoll-Sanchez archive spec, when
    downloaded + extracted, must produce a directory layout that
    ``loader._neuropeptide_new_root()`` recognises as a valid ``new/``
    root. Catches regressions from changing strip_prefix /
    archive_keep_under / extract_to in lockstep.
    """
    # Synthetic archive mirroring the GitHub repo's path structure under
    # whatever wrapper directory GitHub emits.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        wrap = "Neuropeptide-Connectome-abcdef/"
        kept = "Adjacency matrices for networks/"
        # Pairs file — the marker `_neuropeptide_new_root()` looks for.
        z.writestr(
            wrap + kept + "neuropeptide_pairs (network identities for Individual_net folders).csv",
            "ligand,gpcr\nNPP-1,GPCR-1\n",
        )
        # One representative network from each range model.
        for folder, suffix in [
            ("Individual NPP-GPCR networks LR", "long"),
            ("Individual NPP-GPCR networks MR", "mid"),
            ("Individual NPP-GPCR networks SR", "short"),
        ]:
            z.writestr(
                wrap + kept + folder + "/01022024_neuropeptide_network001_" + suffix + "_range_model.csv",
                "0,1\n1,0\n",
            )
        # Sibling directory we explicitly want filtered out.
        z.writestr(wrap + "Scripts & data/ignore.R", "x <- 1\n")
    archive_bytes = buf.getvalue()
    sha = hashlib.sha256(archive_bytes).hexdigest()

    # Re-point the registry at a fresh tmp_path so we don't touch the
    # real ``Ripoll-Sanchez_2023/`` directory if one exists, and patch
    # the sha256 to match our synthetic bytes.
    real = DATASET_REGISTRY["ripoll_sanchez_2023"]
    new_dir = tmp_path / "ripoll"
    real_spec = real.download_specs[0]
    patched_spec = DatasetSpec(
        key=real.key, title=real.title,
        expected_dir=new_dir,
        citation=real.citation, source_url=real.source_url,
        license_note=real.license_note,
        download_specs=(DownloadSpec(
            url=real_spec.url,
            target_relpath=real_spec.target_relpath,
            sha256=sha,  # match the synthetic bytes, not the real archive
            description=real_spec.description,
            extract_to=real_spec.extract_to,
            strip_prefix=real_spec.strip_prefix,
            archive_keep_under=real_spec.archive_keep_under,
        ),),
    )
    monkeypatch.setitem(DATASET_REGISTRY, "ripoll_sanchez_2023", patched_spec)
    _patch_requests_get(monkeypatch, {real_spec.url: archive_bytes})

    download_dataset("ripoll_sanchez_2023", progress=lambda _: None)

    # Patch the loader's prefix path so its `_neuropeptide_data_roots()`
    # generator yields our tmp dir as the first candidate.
    monkeypatch.setattr(loader, "DOWNLOAD_DIR", tmp_path)
    monkeypatch.setattr(loader, "prefix_NP", "ripoll/")

    # The sibling-directory probes inside _neuropeptide_data_roots use
    # TOPDIR.parent / "CeDNe" / ... — patch TOPDIR away from anything real.
    monkeypatch.setattr(loader, "TOPDIR", tmp_path / "nonexistent")

    found = loader._neuropeptide_new_root()
    assert (found / "neuropeptide_pairs (network identities for Individual_net folders).csv").exists()
    assert (found / "Individual NPP-GPCR networks LR" /
            "01022024_neuropeptide_network001_long_range_model.csv").exists()
    assert (found / "Individual NPP-GPCR networks MR" /
            "01022024_neuropeptide_network001_mid_range_model.csv").exists()
    assert (found / "Individual NPP-GPCR networks SR" /
            "01022024_neuropeptide_network001_short_range_model.csv").exists()
    # The sibling directory we filtered out must not be under the new/ tree.
    assert not (found / "Scripts & data").exists()


def test_registered_specs_have_consistent_target_paths():
    """Every DownloadSpec in the real registry must specify a
    target_relpath (avoid relying on URL-tail inference for legacy
    datasets where loaders pin specific filenames). sha256 is encouraged
    but not required for archive datasets where the post-extract files
    aren't individually hashed.
    """
    for key, spec in DATASET_REGISTRY.items():
        for ds in spec.download_specs:
            assert ds.target_relpath, (
                f"Dataset '{key}' spec for {ds.url} has empty target_relpath"
            )
