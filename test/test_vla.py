"""The VLA interface, dataset slicing and chunk scoring."""
import numpy as np
import pytest

from dexter.eval import score_chunks
from dexter.vla import available
from dexter.vla.base import Observation


def test_registry_lists_policies():
    assert "pi05_droid" in available()


def test_budget_is_chunk_duration_not_tick():
    """A chunk policy's deadline is the motion it buys, not the control period."""
    class Fake:
        chunk_size, control_hz = 15, 15.0
        budget_ms = property(lambda self: self.chunk_size / self.control_hz * 1e3)
    assert Fake().budget_ms == pytest.approx(1000.0)


def test_score_rewards_tracking_and_exposes_baseline():
    truth = np.random.default_rng(0).normal(size=(8, 15, 8)) * 0.3
    good = truth + np.random.default_rng(1).normal(size=truth.shape) * 0.02
    s = score_chunks(list(good), list(truth))
    assert s.correlation > 0.9 and s.mae < s.baseline_mae

    # A policy that outputs nothing must not look good just because MAE is low.
    idle = score_chunks(list(np.zeros_like(truth)), list(truth))
    assert np.isnan(idle.correlation) or abs(idle.correlation) < 0.1
    assert idle.mae == pytest.approx(idle.baseline_mae)


def test_observation_keeps_images_native():
    """Resizing at the call site would distort before the model's padded resize."""
    img = np.zeros((180, 320, 3), dtype=np.uint8)
    obs = Observation(images={"cam": img}, state=np.zeros(8, np.float32), task="x")
    assert obs.images["cam"].shape == (180, 320, 3)


def test_slice_only_offers_decodable_episodes(tmp_path):
    """Episodes whose video file was never downloaded must be excluded.

    Sampling one of those silently seeks into the wrong file and returns frames
    from an unrelated episode -- which looked like a bad policy, not a bad read.
    """
    import json
    import pandas as pd
    from dexter.data import LeRobotSlice

    stream = "observation.images.cam"
    (tmp_path / "meta/episodes/chunk-000").mkdir(parents=True)
    (tmp_path / "data/chunk-000").mkdir(parents=True)
    (tmp_path / f"videos/{stream}/chunk-000").mkdir(parents=True)
    (tmp_path / f"videos/{stream}/chunk-000/file-000.mp4").write_bytes(b"")
    (tmp_path / "meta/info.json").write_text(json.dumps({"fps": 15}))
    pd.DataFrame({"episode_index": [0, 1]}).to_parquet(tmp_path / "data/chunk-000/file-000.parquet")
    pd.DataFrame({
        "episode_index": [0, 1],
        f"videos/{stream}/chunk_index": [0, 0],
        f"videos/{stream}/file_index": [0, 7],       # episode 1 lives in a file we lack
        f"videos/{stream}/from_timestamp": [0.0, 900.0],
    }).to_parquet(tmp_path / "meta/episodes/chunk-000/file-000.parquet")

    data = LeRobotSlice(tmp_path, (stream,))
    assert data.episodes.episode_index.tolist() == [0]
