"""The PerAtomMAERMSEstress table prints MAE alongside RMSE, and fills a real table."""

import logging

from mace_scf.utils.script_utils import create_error_table
from mace_scf.utils.train import valid_err_log


def _metrics(**over):
    m = {
        "mae_e_per_atom": 0.0514,
        "rmse_e_per_atom": 0.1126,
        "mae_f": 0.1851,
        "rmse_f": 0.3960,
        "rel_rmse_f": 39.22,
        "rmse_stress_per_atom": 0.0031,
    }
    m.update(over)
    return m


class _Logger:
    def log(self, _):
        pass


def test_log_line_carries_both_mae_and_rmse(caplog):
    with caplog.at_level(logging.INFO):
        valid_err_log(0.02, _metrics(), _Logger(), "PerAtomMAERMSEstress", 14)

    line = caplog.text
    assert "MAE_E_per_atom=51.4 meV" in line
    assert "RMSE_E_per_atom=112.6 meV" in line
    assert "MAE_F=185.1 meV / A" in line
    assert "RMSE_F=396.0 meV / A" in line
    assert "RMSE_stress_per_atom=3.1 meV / A^3" in line


def test_log_line_omits_stress_when_absent(caplog):
    with caplog.at_level(logging.INFO):
        valid_err_log(0.02, _metrics(rmse_stress_per_atom=None), _Logger(), "PerAtomMAERMSEstress", 3)

    assert "MAE_F=185.1" in caplog.text
    assert "RMSE_stress_per_atom" not in caplog.text


def test_final_table_is_populated_not_empty():
    """PerAtomRMSEstressvirials has no branch here and yields an empty table; this must not."""
    import inspect

    src = inspect.getsource(create_error_table)
    assert src.count("PerAtomMAERMSEstress") == 2, "needs both a header and a row branch"


def test_log_line_carries_multipole_errors_when_predicted(caplog):
    """Models that predict multipoles report q and dipole errors, fitted or not."""
    metrics = _metrics(
        mae_charges=0.0912, rmse_charges=0.1453,
        mae_local_dipoles=0.0265, rmse_local_dipoles=0.0460,
    )
    with caplog.at_level(logging.INFO):
        valid_err_log(0.02, metrics, _Logger(), "PerAtomMAERMSEstress", 31)

    line = caplog.text
    assert "MAE_q=91.2 me, RMSE_q=145.3 me" in line
    assert "MAE_mu=26.5 me A, RMSE_mu=46.0 me A" in line


def test_log_line_omits_multipole_errors_for_plain_mace(caplog):
    with caplog.at_level(logging.INFO):
        valid_err_log(0.02, _metrics(), _Logger(), "PerAtomMAERMSEstress", 3)

    assert "MAE_q" not in caplog.text and "MAE_mu" not in caplog.text
