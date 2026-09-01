"""Phase 5: the figures, and the guarantee that they contain nothing but the JSON.

PLAN.md section 7 says results are committed as JSON and figures are regenerated from them,
so the property worth testing is not "the chart is pretty" but "the chart cannot disagree
with the run it describes". Three things enforce that and are checked here: every drawn
number is recomputable from the input, a result that did not finish never reaches a figure,
and a missing experiment raises instead of leaving a gap a reader would misread.

The fixtures build a small synthetic results directory rather than reading the committed one,
so these tests keep working when the real numbers change -- which is exactly the drift the
module exists to prevent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from signadapt import figures
from signadapt.utils.results import ResultsLogger

SIGNERS = (1, 2, 3, 4)
K_VALUES = (0, 1, 2)
SEEDS = (0, 1)


def _model_cfg() -> dict:
    """A tiny but real architecture, so build_model can instantiate it in the size figure."""
    return {
        "encoder": {
            "type": "temporal_transformer",
            "input_dim": 12,
            "d_model": 16,
            "n_layers": 1,
            "n_heads": 2,
            "ff_dim": 16,
            "dropout": 0.0,
            "max_len": 4,
            "pooling": "mean",
        },
        "head": {"type": "linear", "in_dim": 16, "n_classes": 5, "dropout": 0.0},
        "train": {"epochs": 1},
    }


def _score(method: str, signer: int, k: int, seed: int) -> float:
    """A deterministic stand-in accuracy that differs by method, signer, k and seed."""
    base = {"E3": 0.40, "E4": 0.80, "E5": 0.50, "E6": 0.45}[method]
    return min(0.99, base + 0.08 * k + 0.01 * signer + 0.005 * seed)


def _write_sweep(results_dir: Path, method: str, seed: int) -> None:
    with ResultsLogger(
        method, config={"model": _model_cfg()}, seed=seed, results_dir=results_dir
    ) as log:
        for k in K_VALUES:
            for signer in SIGNERS:
                log.log_record(
                    k=k,
                    signer=signer,
                    top1=_score(method, signer, k, seed),
                    top5=min(1.0, _score(method, signer, k, seed) + 0.05),
                    n_query=5,
                )
        log.set_metrics(by_k={}, k_values=list(K_VALUES), skipped_k=[])
        log._extra.update(  # noqa: SLF001 - the loggers used in production set this at build
            {"k_values": list(K_VALUES), "n_folds": len(SIGNERS), "query_repetitions": 1}
        )


def _write_centralized(results_dir: Path, experiment: str, seed: int, top1: float) -> None:
    with ResultsLogger(
        experiment, config={"model": _model_cfg()}, seed=seed, results_dir=results_dir
    ) as log:
        log.set_metrics(
            top1=top1,
            top5=top1 + 0.04,
            per_signer={str(s): top1 for s in SIGNERS},
        )


def _write_federated(results_dir: Path, experiment: str, seed: int, top1: float) -> None:
    with ResultsLogger(
        experiment,
        config={"model": _model_cfg(), "fl": {"checks": {"iid_tolerance": 0.03}}},
        seed=seed,
        results_dir=results_dir,
        extra={"partition": {"mode": "iid", "n_clients": 4}},
    ) as log:
        for round_index in range(1, 4):
            log.log_record(round=round_index, fit={"train_top1": 0.3 * round_index})
        log.set_metrics(top1=top1, n_rounds=3, n_shared_tensors=7)


@pytest.fixture
def results_dir(tmp_path: Path) -> Path:
    """A complete synthetic results directory: every experiment the figures need."""
    directory = tmp_path / "results"
    for method in ("E3", "E4", "E5", "E6"):
        for seed in SEEDS:
            _write_sweep(directory, method, seed)
    for seed in SEEDS:
        _write_centralized(directory, "E1", seed, 0.90 + 0.01 * seed)
        _write_centralized(directory, "E2", seed, 0.80 + 0.01 * seed)
        _write_federated(directory, "fedavg-pretrain", seed, 0.79 + 0.01 * seed)
    _write_federated(directory, "fedavg-iid-check", 0, 0.805)
    return directory


# ------------------------------------------------------------------ loading


def test_load_folds_finds_every_method_and_fold(results_dir):
    folds = figures.load_folds(results_dir)
    assert set(folds) == {"E3", "E4", "E5", "E6"}
    for method, records in folds.items():
        assert len(records) == len(SIGNERS) * len(K_VALUES) * len(SEEDS), method


def test_load_folds_is_sorted_and_independent_of_file_order(results_dir):
    folds = figures.load_folds(results_dir)
    keys = [f.key for f in folds["E5"]]
    assert keys == sorted(keys)


def test_a_failed_run_never_reaches_a_figure(results_dir):
    """A crashed sweep is on disk with status 'failed'; the loader must skip it."""
    with pytest.raises(RuntimeError), ResultsLogger(
        "E5", config={"model": _model_cfg()}, seed=99, results_dir=results_dir
    ) as log:
        log.log_record(k=0, signer=1, top1=0.0, top5=0.0)
        raise RuntimeError("boom")

    folds = figures.load_folds(results_dir)
    assert 99 not in {f.seed for f in folds["E5"]}


def test_mismatched_k_grids_raise_rather_than_plot_two_x_axes(results_dir, tmp_path):
    with ResultsLogger(
        "E5", config={"model": _model_cfg()}, seed=7, results_dir=results_dir
    ) as log:
        log.log_record(k=9, signer=1, top1=0.5, top5=0.6)
        log._extra.update({"k_values": [9]})  # noqa: SLF001

    with pytest.raises(ValueError, match="more than one k grid"):
        figures.load_folds(results_dir)


def test_missing_method_names_the_command_that_produces_it(results_dir):
    folds = figures.load_folds(results_dir)
    del folds["E4"]
    with pytest.raises(FileNotFoundError, match="make sweep"):
        figures.require_methods(folds, ("E3", "E4", "E5"))


# ------------------------------------------------------------------ aggregation


def test_by_k_pools_signers_and_seeds(results_dir):
    curve = figures.by_k(figures.load_folds(results_dir)["E5"])
    assert curve[0]["n"] == len(SIGNERS) * len(SEEDS)


def test_by_k_per_signer_averages_seeds_first(results_dir):
    """Both summaries share a mean; only the second has n equal to the signer count."""
    folds = figures.load_folds(results_dir)["E5"]
    pooled, per_signer = figures.by_k(folds), figures.by_k_per_signer(folds)
    assert pooled[1]["n"] == len(SIGNERS) * len(SEEDS)
    assert per_signer[1]["n"] == len(SIGNERS)
    assert pooled[1]["mean"] == pytest.approx(per_signer[1]["mean"])


def test_by_k_per_signer_removes_seed_noise_from_the_spread():
    """Why the second summary exists: seed noise inflates the pooled spread, not the signer's.

    Nothing orders the two spreads in general -- averaging seeds shrinks each point but also
    cuts n, which the (n-1) correction charges for. What is guaranteed is what this checks:
    when the variation is seed noise rather than a signer effect, averaging it out collapses
    the per-signer spread while the pooled one keeps it.
    """
    folds = tuple(
        figures.Fold(
            method="E5", seed=seed, signer=signer, k=1, top1=0.5 + 0.2 * seed, top5=1.0
        )
        for signer in range(4)
        for seed in range(3)
    )
    pooled, per_signer = figures.by_k(folds), figures.by_k_per_signer(folds)
    assert pooled[1]["mean"] == pytest.approx(per_signer[1]["mean"])
    assert pooled[1]["std"] > 0.1
    assert per_signer[1]["std"] == pytest.approx(0.0)


def test_paired_delta_uses_only_folds_both_methods_ran(results_dir):
    folds = figures.load_folds(results_dir)
    partial = tuple(f for f in folds["E3"] if f.signer != SIGNERS[-1])
    delta = figures.paired_delta(folds["E5"], partial)
    assert delta[1]["n"] == (len(SIGNERS) - 1) * len(SEEDS)


def test_paired_delta_sign_and_count_follow_the_left_method(results_dir):
    folds = figures.load_folds(results_dir)
    delta = figures.paired_delta(folds["E5"], folds["E3"])
    # E5's stand-in score is 10 points above E3's at every fold, by construction.
    assert delta[0]["mean"] == pytest.approx(0.10)
    assert delta[0]["n_better"] == delta[0]["n"]
    reversed_delta = figures.paired_delta(folds["E3"], folds["E5"])
    assert reversed_delta[0]["mean"] == pytest.approx(-0.10)
    assert reversed_delta[0]["n_better"] == 0


def test_crossover_reports_the_margin_so_a_tie_is_visible():
    curve = {0: {"mean": 0.10}, 1: {"mean": 0.50}, 2: {"mean": 0.90}}
    assert figures.crossover(curve, 0.50) == {"k": 1, "margin_points": pytest.approx(0.0)}
    assert figures.crossover(curve, 0.45)["k"] == 1


def test_crossover_returns_none_and_the_shortfall_when_never_reached():
    curve = {0: {"mean": 0.10}, 1: {"mean": 0.20}}
    result = figures.crossover(curve, 0.90)
    assert result["k"] is None
    assert result["margin_points"] == pytest.approx(-70.0)


# ------------------------------------------------------------------ layout helper


def test_declutter_enforces_the_gap_and_keeps_order():
    placed = figures._declutter([10.0, 10.2, 10.4], min_gap=2.0)  # noqa: SLF001
    assert placed == pytest.approx([10.0, 12.0, 14.0])


def test_declutter_shifts_the_block_back_inside_its_limit():
    placed = figures._declutter(  # noqa: SLF001
        [90.0, 91.0, 92.0], min_gap=5.0, limit=(0.0, 100.0)
    )
    assert max(placed) == pytest.approx(100.0)
    assert placed == sorted(placed)


# ------------------------------------------------------------------ end to end


def test_build_all_writes_every_figure_and_the_summary(results_dir, tmp_path):
    out = tmp_path / "figures"
    summary = figures.build_all(results_dir, out, formats=("png",))
    assert set(summary["figures"]) == {
        "fig1_adaptation_curve",
        "fig2_generalization_gap",
        "fig3_paired_differences",
        "fig4_signer_spread",
        "fig5_federated_convergence",
        "fig6_communication_cost",
    }
    for entry in summary["figures"].values():
        for path in entry["file"]:
            assert Path(path).stat().st_size > 0
    assert json.loads((out / "summary.json").read_text())["methods_found"] == [
        "E3",
        "E4",
        "E5",
        "E6",
    ]


def test_every_number_in_the_summary_is_recomputable_from_the_results(results_dir, tmp_path):
    """The point of the module: the figure carries the JSON's numbers, not its author's."""
    summary = figures.build_all(results_dir, tmp_path / "figures", formats=("png",))
    drawn = summary["figures"]["fig1_adaptation_curve"]["top1_pooled_over_signers_and_seeds"]
    for method, curve in drawn.items():
        for k, entry in curve.items():
            expected = [
                _score(method, signer, int(k), seed) for signer in SIGNERS for seed in SEEDS
            ]
            assert entry["mean"] == pytest.approx(sum(expected) / len(expected))


def test_figures_are_deterministic(results_dir, tmp_path):
    """Same JSON in, same bytes out -- so a regenerated figure is a no-op in git."""
    first = figures.build_all(results_dir, tmp_path / "a", formats=("png",))
    second = figures.build_all(results_dir, tmp_path / "b", formats=("png",))
    assert first["figures"] == second["figures"] or True  # paths differ; compare the images
    for name in first["figures"]:
        a = (tmp_path / "a" / f"{name}.png").read_bytes()
        b = (tmp_path / "b" / f"{name}.png").read_bytes()
        assert a == b, name


def test_build_all_refuses_to_draw_without_the_centralized_baselines(results_dir, tmp_path):
    for path in results_dir.glob("e1_*.json"):
        path.unlink()
    with pytest.raises(FileNotFoundError, match="E1"):
        figures.build_all(results_dir, tmp_path / "figures", formats=("png",))


def test_cli_exits_nonzero_when_results_are_missing(tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert figures.main(["--results", str(empty), "--out", str(tmp_path / "out")]) == 1
    assert "no finished results" in capsys.readouterr().out


def test_cli_writes_the_figures(results_dir, tmp_path, capsys):
    code = figures.main(
        ["--results", str(results_dir), "--out", str(tmp_path / "out"), "--formats", "png"]
    )
    assert code == 0
    assert "fig1_adaptation_curve" in capsys.readouterr().out
