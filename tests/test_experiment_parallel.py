import tempfile
import threading
import unittest
from concurrent.futures import Future
from dataclasses import asdict
from unittest.mock import patch

from experiment_parallel import run_experiments_parallel
from experiment_store import load_results_index, save_experiment_result_atomic
from experiment_types import ExperimentResult, ExperimentSpec


class _FakeExecutor:
    def __init__(self, *args, **kwargs):
        self.initializer = kwargs.get("initializer")
        self.initargs = kwargs.get("initargs", ())

    def __enter__(self):
        if self.initializer is not None:
            self.initializer(*self.initargs)
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def submit(self, fn, *args, **kwargs):
        future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as exc:  # pragma: no cover - defensive
            future.set_exception(exc)
        return future


class ExperimentParallelTests(unittest.TestCase):
    def test_parallel_executor_preserves_submission_order(self):
        specs = [
            ExperimentSpec(family="momentum", params={"LOOKBACK_WEEKS": 20}, config_hash="a", experiment_id="a"),
            ExperimentSpec(family="momentum", params={"LOOKBACK_WEEKS": 26}, config_hash="b", experiment_id="b"),
            ExperimentSpec(family="superstock", params={"max_positions": 5}, config_hash="c", experiment_id="c"),
        ]

        def fake_run_single_experiment(*args, **kwargs):
            spec = kwargs["spec"]
            return ExperimentResult(
                spec=spec,
                status="success",
                objective_score=float(ord(spec.experiment_id[0])),
                metrics={"sharpe": 1.0, "calmar": 1.0, "total_return": 1.0, "trades_per_year": 1.0},
                robustness={"viable": True},
                artifacts={},
            )

        with patch("experiment_parallel.ProcessPoolExecutor", _FakeExecutor), patch(
            "experiment_parallel.run_single_experiment", side_effect=fake_run_single_experiment
        ):
            results, worker_failures = run_experiments_parallel(
                specs,
                data={"close": None},
                max_workers=4,
                baseline_by_family={"momentum": "momentum_champion_s10005"},
            )

        self.assertEqual(worker_failures, 0)
        self.assertEqual([result.spec.experiment_id for result in results], ["a", "b", "c"])

    def test_concurrent_result_persistence_is_atomic(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = ExperimentSpec(
                family="momentum",
                params={"LOOKBACK_WEEKS": 26},
                config_hash="hash1",
                experiment_id="exp1",
                timestamp_utc="2026-04-03T00:00:00+00:00",
                benchmark_source="spy_symbol",
            )
            result = ExperimentResult(
                spec=spec,
                status="success",
                objective_score=1.0,
                metrics={"sharpe": 1.0, "calmar": 1.0, "total_return": 10.0, "trades_per_year": 5.0},
                robustness={"viable": True},
                artifacts={},
            )
            payload = asdict(result)
            payload["spec"] = asdict(result.spec)

            barrier = threading.Barrier(2)
            outcomes: list[bool] = []

            def writer():
                barrier.wait()
                outcomes.append(save_experiment_result_atomic(payload, base_dir=tmp))

            threads = [threading.Thread(target=writer) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            self.assertEqual(sum(1 for outcome in outcomes if outcome), 1)
            index = load_results_index(tmp)
            rows = index[(index["config_hash"] == "hash1") & (index["status"] == "success")]
            self.assertEqual(len(rows), 1)


if __name__ == "__main__":
    unittest.main()
