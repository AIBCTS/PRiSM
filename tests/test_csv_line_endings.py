"""CSV writers must emit LF on every platform.

pandas defaults ``to_csv(lineterminator=...)`` to ``os.linesep``, and a text-mode
``open()`` translates ``\\n`` the same way, so an unqualified write produces CRLF on
Windows and LF everywhere else. That matters here because PRiSM hashes the CSVs it
writes: ``save_split_datasets`` records a SHA-256 per split under ``data_provenance``,
and ``data/raw/SHA256SUMS.txt`` records one per raw input. A digest taken from CRLF
bytes cannot be reproduced on Linux or macOS, which silently turns a provenance check
into a platform check.

The companion guard for tracked inputs is ``tests/test_raw_data_checksums.py``; this
module covers the files PRiSM itself writes.
"""

import ast
import hashlib
from pathlib import Path

import pandas as pd
import pytest

import prism
from prism.data_utilities import save_split_datasets
from prism.feature_labels import generate_label_file_template

PRISM_ROOT = Path(prism.__file__).parent


def _to_csv_calls(tree):
    """Yield every ``<something>.to_csv(...)`` Call node in an AST."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == 'to_csv':
                yield node


def test_every_to_csv_call_in_prism_pins_the_line_terminator():
    """A new to_csv call must not silently reintroduce os.linesep.

    Writing to a string buffer (``to_csv()`` with no path) is unaffected by the
    default, but no such call exists in the package today; if one is added, exempt
    it here rather than weakening the rule.
    """
    offenders = []

    for path in sorted(PRISM_ROOT.rglob('*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        for call in _to_csv_calls(tree):
            keywords = {kw.arg for kw in call.keywords}
            if 'lineterminator' not in keywords:
                offenders.append(f"{path.relative_to(PRISM_ROOT.parent)}:{call.lineno}")

    assert not offenders, (
        "to_csv called without lineterminator='\\n' at: "
        + ', '.join(offenders)
        + ". Without it pandas uses os.linesep, so the file is byte-different "
        "(and hash-different) on Windows."
    )


@pytest.fixture
def splits():
    frame = pd.DataFrame({'age': [60, 55, 71, 48], 'target': [0, 1, 1, 0]})
    return frame.iloc[:2], frame.iloc[2:3], frame.iloc[3:]


@pytest.mark.parametrize('header_comment', [None, 'provenance: synthetic'])
def test_save_split_datasets_writes_lf_only(splits, tmp_path, header_comment):
    """Covers both branches: the plain write and the comment-then-append write."""
    train_df, test_df, val_df = splits

    result = save_split_datasets(
        train_df,
        test_df,
        val_df,
        base_filename='lf_check',
        save_dir=tmp_path,
        include_timestamp=False,
        header_comment=header_comment,
    )

    written = sorted(tmp_path.glob('*.csv'))
    assert len(written) == 3

    for path in written:
        raw = path.read_bytes()
        assert b'\r\n' not in raw, f"{path.name} contains CRLF"
        assert b'\r' not in raw, f"{path.name} contains a bare CR"

    # The digests recorded under data_provenance must be the digests of those
    # LF bytes, which is what makes them reproducible on another platform.
    for split in ('train', 'test', 'val'):
        path = Path(result[f'{split}_path'])
        expected = hashlib.sha256(path.read_bytes()).hexdigest()
        assert result[f'{split}_hash_info']['hash'] == expected


def test_generate_label_file_template_writes_lf_only(tmp_path):
    original_df = pd.DataFrame({'id': [1, 2], 'age': [60, 55], 'target': [0, 1]})
    processed_df = pd.DataFrame({'age': [60, 55], 'target': [0, 1]})
    output_path = tmp_path / 'labels.csv'

    generate_label_file_template(
        original_df=original_df,
        processed_df=processed_df,
        target_variable='target',
        id_variable='id',
        output_path=output_path,
    )

    raw = output_path.read_bytes()
    assert b'\r\n' not in raw
    assert b'\r' not in raw
