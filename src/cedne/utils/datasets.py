"""Dataset registry and missing-dataset error for CeDNe loaders.

Each loader in `cedne.utils.loader` reads files staged under
`data_sources/downloads/<dataset_dir>/...`. When a file is missing,
the raw `FileNotFoundError` from pandas / pickle / open is hard to act on
because the message is just a path. The web frontend can't tell the user
which dataset to download or where to put it.

This module groups every external dataset under a stable `dataset_key`
and surfaces a single `MissingDatasetError` that carries a structured
payload (key, expected path, citation, source URL, hint). The web
backend turns that into a 503 with a JSON body the frontend can render
as an actionable banner; programmatic users get the same information
in the exception's message.

Adding a new external dataset:
    1. Add a `DatasetSpec` entry to `DATASET_REGISTRY` keyed by a stable
       `dataset_key` (lowercase, snake_case, matches the directory under
       `data_sources/downloads/`).
    2. Wrap the loader's `pd.read_*` / `open()` calls with
       `require_dataset_file(path, dataset_key)`.

Bundled-with-the-repo files (e.g. `data_sources/Cell_list.pkl`) live
under `DATADIR` directly, not `DOWNLOAD_DIR`, and are not in this
registry — they ship with the source tree and a missing one is a
packaging bug, not a data-staging gap.
"""

__author__ = "Sahil Moza"
__date__ = "2026-05-07"
__license__ = "MIT"

import hashlib
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .config import (
    DOWNLOAD_DIR,
    cook_connectome,
    witvliet_connectome,
    white_connectome,
    fly_wire,
    winding_connectome,
    ciona_connectome,
    pristionchus_pharynx,
    veraszto_connectome,
    brittin_contactome,
    skuhersky_neuropal,
    atanas_whole_brain,
    atanas_link_prefix,
    atanas_links,
    cengen_links,
    lineage,
    prefix_CENGEN,
    prefix_NT,
    prefix_NP,
    prefix_synaptic_weights,
)


@dataclass(frozen=True)
class DownloadSpec:
    """One downloadable file belonging to a `DatasetSpec`.

    `target_relpath` is interpreted relative to ``DatasetSpec.expected_dir``;
    leave empty to derive the filename from the URL's last path segment.
    `sha256` is optional but strongly recommended — when set, downloads are
    verified before being placed (mismatch raises an error and the partial
    file is discarded), and existing files are skipped only if their hash
    already matches. `description` is a short note shown in CLI / log output.

    Archive support: when `extract_to` is not None, the downloaded file is
    treated as a zip archive and unpacked into ``expected_dir / extract_to``
    (use ``""`` for ``expected_dir`` itself). `strip_prefix` drops that
    many leading path components from each archive entry — handy for zips
    that wrap everything in a single top-level directory you don't want
    (Winding 2023's ``Supplementary-Data-S1/`` is the canonical example).
    `archive_keep_under` further filters: only entries whose path starts
    with that string (post-strip_prefix) are extracted, and the prefix
    itself is removed from the placed path. Use this when an archive
    bundles many directories but you only want one — Ripoll-Sanchez's
    GitHub archive is the canonical example: ``strip_prefix=1`` drops the
    ``Neuropeptide-Connectome-<sha>/`` wrapper, then
    ``archive_keep_under="Adjacency matrices for networks/"`` keeps just
    that subtree and discards the rest of the repo.
    `__MACOSX/`-prefixed resource-fork entries are always skipped.
    `extract_keep_archive=False` deletes the .zip after extraction.
    `sha256` is verified on the downloaded archive itself, not on the
    extracted contents.
    """

    url: str
    target_relpath: str = ""
    sha256: Optional[str] = None
    description: Optional[str] = None
    extract_to: Optional[str] = None
    strip_prefix: int = 0
    archive_keep_under: Optional[str] = None
    extract_keep_archive: bool = False


@dataclass(frozen=True)
class DatasetSpec:
    """Static metadata for an external dataset CeDNe loaders depend on."""

    key: str
    title: str
    expected_dir: Path
    citation: Optional[str] = None
    source_url: Optional[str] = None
    license_note: str = (
        "public"  # 'public', 'public-with-attribution', 'restricted: ...'
    )
    download_specs: tuple = ()  # tuple[DownloadSpec, ...]; empty = manual staging only


def _cengen_download_specs() -> tuple:
    """Express CENGEN's four threshold CSVs as DownloadSpec entries.

    The legacy ``download_datasets('cengen')`` path stripped the ``021821_``
    prefix from each URL when naming the local file; we reproduce that
    naming here so the registry is consistent with what the loader expects
    (``DOWNLOAD_DIR/CENGEN/liberal_threshold1.csv`` etc).

    sha256 hashes were computed from a known-good local copy on
    2026-05-07; if the upstream tables are revised, these will start
    failing and we'll need to update them (CENGEN versions threshold
    tables by date — the ``021821_`` prefix in the URL is the cut date).
    """
    sha256_by_filename = {
        "liberal_threshold1.csv": "74ac7e400f3f841d8784946db67bdbc211598c8682020ef57730b29e564e0cb6",
        "medium_threshold2.csv": "de03fc03cacdf4c1aa7ffd7565df89135ed72eab68d0b3cf03f542932abc63c8",
        "conservative_threshold3.csv": "2454624b7b0409b8ce62c5687c431e0c9b9dd5f38148e0bc9b0fb48c26887236",
        "stringent_threshold4.csv": "478791be7f43d46e11cb751cf12016ae517d59b34db3325f29703646853df875",
    }
    out = []
    for url in cengen_links:
        local_name = url.split("/")[-1].split("021821_")[-1]
        out.append(
            DownloadSpec(
                url=url,
                target_relpath=local_name,
                sha256=sha256_by_filename.get(local_name),
                description=f"CENGEN threshold table ({local_name})",
            )
        )
    return tuple(out)


_WORMWIRING_SI_BASE = "https://wormwiring.org/si/"


def _wormwiring(filename: str) -> str:
    """URL for a WormWiring-hosted supplementary file. They host filenames
    verbatim (with spaces and commas), so we URL-encode here rather than
    in every spec literal.
    """
    from urllib.parse import quote

    return _WORMWIRING_SI_BASE + quote(filename)


def _cook_2019_download_specs() -> tuple:
    """Cook 2019 SI files hosted at WormWiring. The loader's male path
    only opens 'SI 5 Connectome adjacency matrices, corrected July 2020',
    but the other SI files are useful companions (cell lists, synapse
    lists) so we register the full set.

    sha256 hashes computed from a known-good local copy on 2026-05-07.
    """
    fileinfo = [
        # (filename, sha256, description)
        (
            "SI 2 Synapse adjacency matrices.xlsx",
            "ae59aa8b4bdf8f12c3bf14bfb19e71225c3db0dbe42d9155e1f0bdf796d8f2e0",
            "per-synapse adjacency matrices",
        ),
        (
            "SI 3 Synapse lists.xlsx",
            "134e41df6a6159f79edd3514bfa37930c74646abc7f0b52bf07cbc1ccf9e17d8",
            "synapse lists",
        ),
        (
            "SI 4 Cell lists.xlsx",
            "f5c524407e196ba8045cfb86fb996d41730e4b899a7c24b0c525a4d2ad0a7376",
            "cell lists",
        ),
        (
            "SI 5 Connectome adjacency matrices, corrected July 2020.xlsx",
            "1f4fdbf84746b69b49a8da0816f52787860ce349b638dce37924ba80f90c70c9",
            "connectome adjacency matrices (Cook male loader reads this)",
        ),
        (
            "SI 6 Cell class lists.xlsx",
            "6a9d4d5f0568944a04b6f3b42959196bf9c2440dfc5fea4d805f195aaa3195bc",
            "cell class lists",
        ),
        (
            "SI 7 Cell class connectome adjacency matrices, corrected July 2020.xlsx",
            "df2ea697eb9f8805184c1f8a19ff6b69aad5a8d2346814fdb627493aec1f3376",
            "cell-class connectome adjacency matrices",
        ),
    ]
    return tuple(
        DownloadSpec(
            url=_wormwiring(name),
            target_relpath=name,
            sha256=sha,
            description=f"Cook 2019 — {desc}",
        )
        for (name, sha, desc) in fileinfo
    )


def _atanas_download_specs() -> tuple:
    """Express the Atanas 2023 JSON downloads as DownloadSpec entries.

    Files are split across ``Control/`` and ``Heat/`` subdirectories of
    ``expected_dir`` (= ``DOWNLOAD_DIR/Atanas_2023``).
    """
    out = []
    for condition, filenames in atanas_links.items():
        for fname in filenames:
            out.append(
                DownloadSpec(
                    url=atanas_link_prefix + fname,
                    target_relpath=f"{condition}/{fname}",
                    description=f"Atanas {condition} session ({fname})",
                )
            )
    return tuple(out)


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "cengen": DatasetSpec(
        key="cengen",
        title="CeNGEN — single-cell C. elegans transcriptome thresholds",
        expected_dir=DOWNLOAD_DIR / prefix_CENGEN.rstrip("/"),
        citation="Taylor et al. (2021) Cell 184:4329–4347; Taylor et al. (2025) bioRxiv",
        source_url="https://cengen.org/",
        download_specs=_cengen_download_specs(),
    ),
    "cook_2019": DatasetSpec(
        key="cook_2019",
        title="Cook 2019 — C. elegans whole-animal connectome (hermaphrodite + male)",
        expected_dir=cook_connectome,
        citation="Cook et al. (2019) Nature 571:63–71",
        source_url="https://doi.org/10.1038/s41586-019-1352-7",
        download_specs=_cook_2019_download_specs(),
    ),
    "witvliet_2020": DatasetSpec(
        key="witvliet_2020",
        title="Witvliet 2020 — C. elegans connectomes across postembryonic development",
        expected_dir=witvliet_connectome,
        citation="Witvliet et al. (2021) Nature 596:257–261",
        source_url="https://doi.org/10.1038/s41586-021-03778-8",
    ),
    "white_1986": DatasetSpec(
        key="white_1986",
        title="White 1986 — C. elegans original electron-microscopy reconstruction",
        expected_dir=white_connectome,
        citation="White et al. (1986) Phil. Trans. R. Soc. B 314:1–340",
        source_url="https://doi.org/10.1098/rstb.1986.0056",
    ),
    "fly_wire": DatasetSpec(
        key="fly_wire",
        title="FlyWire — Drosophila adult-brain connectome (Codex export)",
        expected_dir=fly_wire,
        citation="Dorkenwald et al. (2024) Nature 634:124–138",
        source_url="https://codex.flywire.ai/",
    ),
    "winding_2023": DatasetSpec(
        key="winding_2023",
        title="Winding 2023 — Drosophila larva whole-brain connectome",
        expected_dir=winding_connectome,
        citation="Winding et al. (2023) Science 379:eadd9330",
        source_url="https://doi.org/10.1126/science.add9330",
        download_specs=(
            DownloadSpec(
                # The authors' GitHub repo packages every CSV (annotations,
                # connectivity matrices, IO ratios, inputs/outputs) as a
                # single zip whose top-level dir is "Supplementary-Data-S1".
                # We strip that prefix on extraction so files land flat in
                # `winding_connectome/` where the loader expects them.
                url="https://raw.githubusercontent.com/brain-networks/larval-drosophila-connectome/main/Supplementary-Data-S1.zip",
                target_relpath="Supplementary-Data-S1.zip",
                sha256="8c1f43809ed5d527ba61b154e377cc21da26383a75eda8aab85ce05607a72a4c",
                description="Winding 2023 — supplementary data archive (10 CSVs)",
                extract_to="",
                strip_prefix=1,
            ),
        ),
    ),
    "ryan_2016": DatasetSpec(
        key="ryan_2016",
        title="Ryan 2016 — Ciona intestinalis larval connectome",
        expected_dir=ciona_connectome,
        citation="Ryan et al. (2016) eLife 5:e16962",
        source_url="https://doi.org/10.7554/eLife.16962",
    ),
    "bumbarger_2013": DatasetSpec(
        key="bumbarger_2013",
        title="Bumbarger 2013 — Pristionchus pacificus pharyngeal connectome",
        expected_dir=pristionchus_pharynx,
        citation="Bumbarger et al. (2013) Cell 152:109–119",
        source_url="https://doi.org/10.1016/j.cell.2012.12.013",
    ),
    "veraszto_2025": DatasetSpec(
        key="veraszto_2025",
        title="Veraszto 2025 — Platynereis dumerilii whole-body 3-day-larva connectome",
        expected_dir=veraszto_connectome,
        citation="Veraszto et al. (2025) eLife 13:e97964",
        source_url="https://doi.org/10.7554/eLife.97964",
    ),
    "brittin_2018": DatasetSpec(
        key="brittin_2018",
        title="Brittin 2018 — C. elegans nerve-ring contactome (adult + L4)",
        expected_dir=brittin_contactome,
        citation="Brittin et al. (2021) Nature 591:105–110",
        source_url="https://doi.org/10.1038/s41586-021-03284-x",
        download_specs=(
            DownloadSpec(
                # Hosted alongside the Cook SI files on WormWiring
                # (the WormWiring SI page lists this file under "Adjacency
                # Matrices" too).
                url=_wormwiring("Adult and L4 nerve ring neighbors.xlsx"),
                target_relpath="Adult and L4 nerve ring neighbors.xlsx",
                sha256="11e1ad438bffd1b0a7727639f55e70f0c7fa44a6859a9dbe775cc67df0819dd5",
                description="Brittin 2018 — adult + L4 nerve-ring contactome",
            ),
        ),
    ),
    "skuhersky_2022": DatasetSpec(
        key="skuhersky_2022",
        title="Skuhersky 2022 — NeuroPAL-derived C. elegans 3D anatomical atlas",
        expected_dir=skuhersky_neuropal,
        citation="Skuhersky et al. (2022) BMC Bioinformatics 23:208",
        source_url="https://doi.org/10.1186/s12859-022-04738-3",
    ),
    "ripoll_sanchez_2023": DatasetSpec(
        key="ripoll_sanchez_2023",
        title="Ripoll-Sanchez 2023 — C. elegans neuropeptide GPCR networks",
        expected_dir=DOWNLOAD_DIR / prefix_NP.rstrip("/"),
        citation="Ripoll-Sanchez et al. (2023) Neuron 111:3570–3589",
        source_url="https://doi.org/10.1016/j.neuron.2023.09.043",
        download_specs=(
            DownloadSpec(
                # The author's GitHub repo is the canonical source; we pin
                # to a commit SHA so URL + sha256 stay reproducible across
                # runs (GitHub re-generates archive zips for branch refs
                # but commit-SHA refs are stable). The ``new/`` format
                # (per-network CSVs in three range models) is what
                # `loadNeuropeptides(mode='new')` reads.
                #
                # The archive bundles the entire repo (~80 MB of scripts,
                # figures, sensitivity-analysis data). `archive_keep_under`
                # filters extraction to just the directory we need so we
                # don't pollute the data PVC with R scripts and PDFs.
                #
                # If upstream advances `main`, update both the URL's
                # commit SHA and the sha256 below in lockstep.
                url="https://github.com/LidiaRipollSanchez/Neuropeptide-Connectome/archive/6689619236ba1b4681a9a77b3d918d513416336c.zip",
                target_relpath="_archive.zip",
                sha256="4f2ac72c546e8129e72f8b86206a8256367bb2fca1ec908cc4441d6c341261e4",
                description="Ripoll-Sanchez 2023 — Neuropeptide-Connectome repo @ 66896192",
                extract_to="new",
                strip_prefix=1,  # drops 'Neuropeptide-Connectome-<sha>/'
                archive_keep_under="Adjacency matrices for networks/",
            ),
        ),
    ),
    "randi_2023": DatasetSpec(
        key="randi_2023",
        title="Randi 2023 — C. elegans signal-propagation atlas (synaptic weights)",
        expected_dir=DOWNLOAD_DIR / prefix_synaptic_weights.rstrip("/"),
        citation="Randi et al. (2023) Nature 623:406–414",
        source_url="https://doi.org/10.1038/s41586-023-06683-4",
        download_specs=(
            DownloadSpec(
                # Nature's static-content host serves SI MOESM files at a
                # deterministic path; the loader reads MOESM13_ESM.xls
                # specifically (the synaptic-weight matrix).
                url="https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-023-06683-4/MediaObjects/41586_2023_6683_MOESM13_ESM.xls",
                target_relpath="41586_2023_6683_MOESM13_ESM.xls",
                sha256="ceb1e279acbb87c0975a4d0fa0d46a7b4d0656acf4906388b61e7a86c66dcd6b",
                description="Randi 2023 — signal-propagation atlas synaptic weights (MOESM13)",
            ),
        ),
    ),
    "wang_2024": DatasetSpec(
        key="wang_2024",
        title="Wang 2024 — C. elegans neurotransmitter ligand calls",
        expected_dir=DOWNLOAD_DIR / prefix_NT.rstrip("/"),
        citation="Wang et al. (2024) eLife 13:RP95402",
        source_url="https://doi.org/10.7554/eLife.95402",
    ),
    "atanas_2023": DatasetSpec(
        key="atanas_2023",
        title="Atanas 2023 — C. elegans whole-brain calcium imaging recordings",
        # Atanas is split across Control / Heat subfolders; expose the parent.
        expected_dir=next(iter(atanas_whole_brain.values())).parent,
        citation="Atanas et al. (2023) Cell 186:4134–4151",
        source_url="https://doi.org/10.1016/j.cell.2023.07.035",
        download_specs=_atanas_download_specs(),
    ),
    "worm_atlas_lineage": DatasetSpec(
        key="worm_atlas_lineage",
        title="WormAtlas — Altun developmental-lineage tables",
        expected_dir=lineage.parent,
        citation="Sulston & Horvitz (1977) Dev. Biol. 56:110–156",
        source_url="https://www.wormatlas.org/",
    ),
}


def _sha256_of(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Stream-hash a file. Used both for download verification and for
    seeding the registry from a known-good local copy.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _stream_download(url: str, dest: Path, *, chunk_size: int = 1 << 20) -> str:
    """Stream-download `url` to `dest`, returning the sha256 of the bytes
    written. Atomic: writes to a sibling temp file, moves on success,
    cleans up on failure.

    Imported here (not at module top) so importing `datasets` doesn't
    require `requests` for non-download code paths (the loaders already
    pin requests, but library users running just `MissingDatasetError`
    pathways shouldn't pay for the import).
    """
    import requests  # local import keeps `datasets` importable without it

    dest.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    fd, tmp_name = tempfile.mkstemp(prefix=dest.name + ".", dir=str(dest.parent))
    tmp_path = Path(tmp_name)
    try:
        with (
            os.fdopen(fd, "wb") as out,
            requests.get(url, stream=True, timeout=60) as resp,
        ):
            resp.raise_for_status()
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                h.update(chunk)
                out.write(chunk)
        shutil.move(str(tmp_path), str(dest))
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise
    return h.hexdigest()


def _extract_zip(
    archive: Path,
    dest_dir: Path,
    *,
    strip_prefix: int = 0,
    archive_keep_under: Optional[str] = None,
) -> list:
    """Extract `archive` into `dest_dir`. Returns the list of files written.

    Skips macOS ``__MACOSX/`` resource-fork entries and ``.DS_Store``.
    Filtering pipeline (in order):
      1. ``strip_prefix`` drops that many leading path components.
      2. ``archive_keep_under`` (if set) keeps only entries whose stripped
         path starts with that string, and removes that prefix from the
         placed path.
      3. The result is placed under ``dest_dir``.

    Refuses entries whose stripped path would escape `dest_dir` (defensive
    against malicious zips, though all of CeDNe's sources are trusted;
    sha256 verification on the archive is the primary guard).
    """
    written = []
    dest_dir = Path(dest_dir).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    keep_under = archive_keep_under or ""
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            name = info.filename
            if name.startswith("__MACOSX/") or name.endswith("/.DS_Store"):
                continue
            parts = name.split("/")
            if strip_prefix:
                if len(parts) <= strip_prefix:
                    continue  # nothing left after stripping
                parts = parts[strip_prefix:]
            stripped = "/".join(parts)
            if keep_under:
                if not stripped.startswith(keep_under):
                    continue
                stripped = stripped[len(keep_under) :]
            if not stripped or stripped.endswith("/"):
                continue  # directory entry; skip (mkdir is implicit on file write)
            target = (dest_dir / stripped).resolve()
            try:
                target.relative_to(dest_dir)
            except ValueError:
                raise RuntimeError(
                    f"Refusing to extract entry '{name}' to '{target}' "
                    f"(escapes destination '{dest_dir}')."
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
            written.append(target)
    return written


@dataclass
class DownloadResult:
    """Per-file outcome of a download_dataset() call."""

    spec: DownloadSpec
    target: Path
    status: str  # 'downloaded' | 'skipped' | 'failed' | 'extracted'
    sha256: Optional[str] = None  # actual sha256 of the file on disk after the op
    error: Optional[str] = None
    extracted_files: Optional[list] = None  # paths written by archive extraction


def download_dataset(
    key: str,
    *,
    force: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> list:
    """Materialise every file in `DATASET_REGISTRY[key].download_specs`.

    Parameters
    ----------
    key:
        Dataset key. Must be in `DATASET_REGISTRY`. ``KeyError`` if not.
    force:
        Re-download files that already exist on disk. Without this, a file
        already present is skipped if its sha256 matches the spec (or if no
        sha256 is registered, it's skipped on path-existence).
    progress:
        Optional one-line logger callback ``progress(msg: str)``. Defaults to
        ``print`` so CLI users see live feedback. Pass ``lambda _: None`` to
        silence output.

    Returns
    -------
    list[DownloadResult]
        One entry per spec, in registry order. Inspect ``status`` for
        ``'downloaded' | 'skipped' | 'failed'`` and ``error`` for failures.
    """
    if key not in DATASET_REGISTRY:
        raise KeyError(
            f"Unknown dataset key '{key}'. " f"Known: {sorted(DATASET_REGISTRY)}"
        )
    spec = DATASET_REGISTRY[key]
    if not spec.download_specs:
        raise ValueError(
            f"Dataset '{key}' has no registered download URLs. "
            f"Stage the data manually under {spec.expected_dir} "
            f"(see {spec.source_url or 'the citation'})."
        )
    log = progress if progress is not None else print
    log(f"==> Downloading dataset '{key}' to {spec.expected_dir}")
    results = []
    for ds in spec.download_specs:
        target = spec.expected_dir / (ds.target_relpath or ds.url.rsplit("/", 1)[-1])
        is_archive = ds.extract_to is not None

        # Skip-when-present is keyed on the archive file when we're
        # extracting (sha256 still applies); otherwise on the target file.
        if not force and target.exists():
            actual = _sha256_of(target) if ds.sha256 else None
            if ds.sha256 and actual == ds.sha256:
                log(
                    f"  [skip]   {target.relative_to(spec.expected_dir)} (sha256 matches)"
                )
                results.append(DownloadResult(ds, target, "skipped", sha256=actual))
                continue
            if ds.sha256 is None:
                log(
                    f"  [skip]   {target.relative_to(spec.expected_dir)} (already present, no sha256 registered)"
                )
                results.append(DownloadResult(ds, target, "skipped"))
                continue
            log(
                f"  [retry]  {target.relative_to(spec.expected_dir)} (existing sha256 differs)"
            )
        try:
            log(f"  [fetch]  {ds.url}")
            actual = _stream_download(ds.url, target)
            if ds.sha256 and actual != ds.sha256:
                target.unlink(missing_ok=True)
                msg = f"sha256 mismatch: expected {ds.sha256}, got {actual}"
                log(f"  [FAIL]   {ds.url}: {msg}")
                results.append(DownloadResult(ds, target, "failed", error=msg))
                continue
            if is_archive:
                extract_dir = spec.expected_dir / ds.extract_to
                written = _extract_zip(
                    target,
                    extract_dir,
                    strip_prefix=ds.strip_prefix,
                    archive_keep_under=ds.archive_keep_under,
                )
                if not ds.extract_keep_archive:
                    target.unlink(missing_ok=True)
                log(f"  [extract] {len(written)} file(s) into {extract_dir}")
                results.append(
                    DownloadResult(
                        ds,
                        target,
                        "extracted",
                        sha256=actual,
                        extracted_files=written,
                    )
                )
            else:
                results.append(DownloadResult(ds, target, "downloaded", sha256=actual))
        except Exception as e:
            log(f"  [FAIL]   {ds.url}: {e}")
            results.append(
                DownloadResult(ds, target, "failed", sha256=None, error=str(e))
            )
    n_dl = sum(1 for r in results if r.status in ("downloaded", "extracted"))
    n_sk = sum(1 for r in results if r.status == "skipped")
    n_fl = sum(1 for r in results if r.status == "failed")
    log(f"==> '{key}': {n_dl} downloaded, {n_sk} skipped, {n_fl} failed")
    return results


def download_all_public(
    *,
    force: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> dict:
    """Iterate over every registered dataset that has download_specs and
    fetch all of them. Datasets without specs are reported as skipped with
    a hint pointing at their `source_url`.

    Returns ``{key: list[DownloadResult]}``. Failures within a dataset
    don't abort the loop — the caller decides what to do with the summary.
    """
    log = progress if progress is not None else print
    summary = {}
    for key, spec in DATASET_REGISTRY.items():
        if not spec.download_specs:
            log(
                f"==> Skipping '{key}': no registered download URLs (obtain from {spec.source_url or 'the citation'})"
            )
            summary[key] = []
            continue
        try:
            summary[key] = download_dataset(key, force=force, progress=log)
        except Exception as e:
            log(f"==> '{key}': raised {type(e).__name__}: {e}")
            summary[key] = []
    return summary


class MergedNetworkError(RuntimeError):
    """Raised by property loaders when called on a network containing
    merged neurons without an explicit aggregation choice.

    Property loaders (``loadTranscripts``, ``loadNeurotransmitters``,
    …) key their data by the *current* neuron name. After
    ``contract_neurons`` runs, a merged neuron like ``AVA_LR`` matches
    no source-data row, so the loader either crashes or silently
    leaves the merged neuron without the loaded property — a quiet
    correctness bug. Callers must either:

      * reload the network and re-merge after loading, or
      * pass ``aggregate=True`` to opt into a per-loader policy that
        unions the loaded property across constituents (with provenance
        recorded as ``_aggregated_from`` on the merged neuron).

    Carries ``merged_names`` so the web backend can surface them in a
    structured 409 response that the UI renders as a confirm dialog.
    """

    def __init__(self, merged_names, *, op_name: str):
        self.merged_names = list(merged_names)
        self.op_name = op_name
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        n = len(self.merged_names)
        sample = ", ".join(self.merged_names[:5])
        if n > 5:
            sample += f", … ({n - 5} more)"
        return (
            f"{self.op_name} cannot run on a network with {n} merged "
            f"neuron{'' if n == 1 else 's'} ({sample}). "
            "Reload the network and contract after loading, or pass "
            "aggregate=True to union the property across constituents "
            "(with provenance recorded on each merged neuron)."
        )

    def to_payload(self) -> dict:
        """JSON-friendly structured representation for API responses."""
        return {
            "error": "merged_network",
            "op_name": self.op_name,
            "merged_names": self.merged_names,
            "message": self._format_message(),
        }


class MissingDatasetError(FileNotFoundError):
    """Raised when a CeDNe loader can't find a required dataset on disk.

    Carries a `dataset_key` so callers (notably the FastAPI backend) can
    format an actionable response instead of a raw `[Errno 2]`. The
    structured payload is exposed via `to_payload()` for serialization.

    Inherits from `FileNotFoundError` so existing `except FileNotFoundError`
    fallbacks (e.g. the optional ligand-table read in
    `loadNeurotransmitters`) continue to swallow the error rather than
    crashing the loader.
    """

    def __init__(
        self,
        dataset_key: str,
        expected_path,
        hint: Optional[str] = None,
    ):
        self.dataset_key = dataset_key
        self.expected_path = Path(expected_path)
        spec = DATASET_REGISTRY.get(dataset_key)
        self.title = spec.title if spec else dataset_key
        self.citation = spec.citation if spec else None
        self.source_url = spec.source_url if spec else None
        self.license_note = spec.license_note if spec else "public"
        self.hint = hint
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        lines = [
            f"Dataset '{self.dataset_key}' is required but not available.",
            f"  Expected at: {self.expected_path}",
        ]
        if self.title and self.title != self.dataset_key:
            lines.append(f"  Source:      {self.title}")
        if self.citation:
            lines.append(f"  Citation:    {self.citation}")
        if self.source_url:
            lines.append(f"  Obtain from: {self.source_url}")
        if self.hint:
            lines.append(f"  Hint:        {self.hint}")
        return "\n".join(lines)

    def to_payload(self) -> dict:
        """JSON-friendly structured representation for API responses."""
        return {
            "error": "missing_dataset",
            "dataset_key": self.dataset_key,
            "title": self.title,
            "expected_path": str(self.expected_path),
            "citation": self.citation,
            "source_url": self.source_url,
            "license_note": self.license_note,
            "hint": self.hint,
            "message": self._format_message(),
        }


def require_dataset_file(path, dataset_key: str, hint: Optional[str] = None) -> Path:
    """Return `Path(path)` if it exists; raise `MissingDatasetError` otherwise.

    Use at the boundary of every loader read so the failure mode is a
    single, structured exception type rather than a path-only
    `FileNotFoundError`.
    """
    p = Path(path)
    if not p.exists():
        raise MissingDatasetError(dataset_key=dataset_key, expected_path=p, hint=hint)
    return p


__all__ = [
    "DatasetSpec",
    "DownloadSpec",
    "DownloadResult",
    "DATASET_REGISTRY",
    "MissingDatasetError",
    "MergedNetworkError",
    "require_dataset_file",
    "download_dataset",
    "download_all_public",
]
