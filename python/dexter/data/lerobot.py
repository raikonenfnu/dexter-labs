"""Read a downloaded slice of a LeRobot v3 dataset.

Datasets like DROID are far too large to fetch whole, so the engine works from
a *slice*: the metadata, one data shard, and one video file per camera. The one
subtlety that matters is that those three do not cover the same episodes --
`data/chunk-000/file-000.parquet` indexes episodes whose frames live in video
files that were never downloaded. Sampling one of those fails deep inside a
video seek, so :class:`LeRobotSlice` resolves availability up front and only
ever offers episodes it can actually decode.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Sample:
    """One prediction point: what the robot saw, and what the operator did."""

    episode: int
    offset: int
    images: dict[str, np.ndarray]   # stream name -> HxWx3 uint8
    frames: "object"                # the underlying rows, for column access
    task: str


class LeRobotSlice:
    """Random access into a partially-downloaded LeRobot dataset."""

    def __init__(self, root: str | Path, streams: tuple[str, ...]) -> None:
        self.root = Path(root)
        self.streams = streams
        self.info = json.loads((self.root / "meta/info.json").read_text())
        self.fps: int = self.info["fps"]
        self._readers: dict[str, object] = {}

    @cached_property
    def frames(self):
        import pandas as pd

        return pd.read_parquet(self.root / "data/chunk-000/file-000.parquet")

    @cached_property
    def episodes(self):
        """Episodes that are fully decodable from what was downloaded.

        An episode qualifies only if its frames are in the data shard *and*
        every camera it needs points at a video file present on disk.
        """
        import pandas as pd

        meta = pd.read_parquet(self.root / "meta/episodes/chunk-000/file-000.parquet")
        meta = meta[meta.episode_index.isin(set(self.frames.episode_index.unique()))]
        for stream in self.streams:
            chunk = meta[f"videos/{stream}/chunk_index"]
            file = meta[f"videos/{stream}/file_index"]
            available = [
                (c, f) for c, f in {(int(a), int(b)) for a, b in zip(chunk, file)}
                if (self.root / f"videos/{stream}/chunk-{c:03d}/file-{f:03d}.mp4").exists()
            ]
            keep = [(c, f) in available for c, f in zip(chunk, file)]
            meta = meta[keep]
        return meta.reset_index(drop=True)

    def _decode(self, stream: str, timestamp: float) -> np.ndarray:
        import av

        container = self._readers.get(stream)
        if container is None:
            path = self.root / f"videos/{stream}/chunk-000/file-000.mp4"
            container = av.open(str(path))
            self._readers[stream] = container

        video = container.streams.video[0]
        target = int(timestamp / float(video.time_base))
        container.seek(target, stream=video, any_frame=False)
        for frame in container.decode(video=0):
            if frame.pts is not None and frame.pts >= target:
                return frame.to_ndarray(format="rgb24")
        raise RuntimeError(f"{stream}: no frame at {timestamp:.2f}s")

    def sample(self, episode: int, offset: int, length: int) -> Sample | None:
        """Frames and rows for ``length`` steps starting at ``offset``."""
        meta = self.episodes[self.episodes.episode_index == episode]
        if meta.empty:
            return None
        meta = meta.iloc[0]
        rows = self.frames[self.frames.episode_index == episode].iloc[offset : offset + length]
        if len(rows) < length:
            return None

        images = {
            stream: self._decode(stream,
                                 float(meta[f"videos/{stream}/from_timestamp"]) + offset / self.fps)
            for stream in self.streams
        }
        task = str(rows.iloc[0].get("language_instruction") or "")
        return Sample(episode, offset, images, rows, task)

    def iter_samples(self, length: int, *, count: int, seed: int = 0,
                     accept=None):
        """Yield up to ``count`` random samples that ``accept`` approves."""
        rng = np.random.default_rng(seed)
        episodes = self.episodes.episode_index.to_numpy()
        yielded = attempts = 0
        while yielded < count and attempts < count * 20:
            attempts += 1
            episode = int(rng.choice(episodes))
            rows = self.frames[self.frames.episode_index == episode]
            if len(rows) < length + 2:
                continue
            sample = self.sample(episode, int(rng.integers(0, len(rows) - length)), length)
            if sample is None or (accept is not None and not accept(sample)):
                continue
            yielded += 1
            yield sample
