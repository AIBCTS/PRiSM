"""Checks that the recorded raw-data checksums still describe the tracked files.

``data/raw/SHA256SUMS.txt`` is the provenance record for the raw datasets. It is
only worth anything if it verifies on every checkout, which is why these tests
exist: the digests were once recorded on Windows, where git's autocrlf smudge
had rewritten every LF to CRLF, so three of the four entries verified on no
other platform. ``.gitattributes`` now pins tracked CSVs to LF; the tests below
fail if either half of that arrangement is undone.

Both tests skip rather than fail when their inputs are absent. The suite runs
against three different trees -- this repository, the public repository (which
carries only htx_example), and an installed wheel (which carries no data/raw at
all) -- and a missing file means "not this tier", not "broken".
"""

import hashlib
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SHA256SUMS = REPO_ROOT / 'data' / 'raw' / 'SHA256SUMS.txt'


def parse_sha256sums(text):
    """Parse coreutils sha256sum output into (filename, digest) pairs.

    Accepts only the two-space text form, which is what publish_public.py's
    filtered_sha256sums() expects when it splits a line to decide whether the
    named file is published. The binary form that some sha256sum builds emit by
    default ('<digest> *name') would leave the asterisk stuck to the filename
    and silently drop the entry from the published manifest.
    """
    entries = []
    for line in text.splitlines():
        if not line.strip():
            continue
        digest, _, name = line.partition('  ')
        assert digest and name, f"malformed checksum line: {line!r}"
        entries.append((name, digest))
    return entries


@pytest.mark.skipif(not SHA256SUMS.exists(), reason="no data/raw in this tree")
def test_recorded_digests_match_the_files_on_disk():
    entries = parse_sha256sums(SHA256SUMS.read_text(encoding='utf-8'))
    assert entries, "SHA256SUMS.txt is empty"

    checked = 0
    for name, expected in entries:
        path = SHA256SUMS.parent / name
        if not path.exists():
            continue  # dataset not present in this tier
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == expected, (
            f"{name} does not match its recorded digest. If the file was "
            f"regenerated, refresh SHA256SUMS.txt. If it was not, the checkout "
            f"has altered the bytes -- check that .gitattributes still pins "
            f"CSVs to eol=lf."
        )
        checked += 1

    if checked == 0:
        pytest.skip("none of the listed datasets are present in this tree")


def tracked_csvs():
    """Paths of the CSVs git tracks, or None outside a git checkout."""
    try:
        result = subprocess.run(
            ['git', 'ls-files', '-z', '--', '*.csv'],
            cwd=REPO_ROOT,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    names = [n for n in result.stdout.decode('utf-8').split('\0') if n]
    return [REPO_ROOT / n for n in names]


def test_no_tracked_csv_is_checked_out_with_crlf():
    """A CRLF here means the eol=lf attribute stopped applying.

    Every digest recorded for a CSV -- in SHA256SUMS.txt, and in the
    data_provenance block preprocess_data() writes -- is a digest of the bytes
    on disk, so line endings that vary by platform make those digests
    unverifiable anywhere but the machine that produced them.
    """
    paths = tracked_csvs()
    if paths is None:
        pytest.skip("not a git checkout")
    present = [p for p in paths if p.exists()]
    if not present:
        pytest.skip("no tracked CSVs in this tree")

    offenders = [str(p.relative_to(REPO_ROOT)) for p in present if b'\r\n' in p.read_bytes()]
    assert not offenders, (
        "tracked CSVs checked out with CRLF line endings: "
        f"{offenders}. Confirm .gitattributes still carries '*.csv text eol=lf', "
        "then force a fresh checkout of those files."
    )
