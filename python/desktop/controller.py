"""Single-owner worker with bounded desktop progress messages."""

from dataclasses import dataclass
import json
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread

from cfdSolver.torchIO import TemporalWriter


@dataclass
class Progress:
    time: float
    steps: int
    frame: Path


@dataclass
class Result:
    session: object
    error: str = ""


class SessionRunner:
    """Pause between steps and keep at most one pending UI update."""

    def __init__(self):
        self.progress = Queue(maxsize=1)
        self.results = Queue(maxsize=1)
        self.pause = Event()
        self.thread = None
        self.finished = Event()
        self.finished.set()

    @property
    def running(self):
        return self.thread is not None and not self.finished.is_set()

    def start(self, session, prefix, interval):
        if self.running:
            raise RuntimeError("A simulation is already running")
        if not isinstance(interval, int) or interval < 1:
            raise ValueError("Recording interval must be a positive integer")
        if self.thread is not None:
            self.thread.join()
        self.finished.clear()
        for queue in (self.progress, self.results):
            while not queue.empty():
                queue.get_nowait()
        self.pause.clear()
        self.thread = Thread(
            target=self._work,
            args=(session, Path(prefix), interval),
            daemon=True,
        )
        self.thread.start()

    def request_pause(self):
        self.pause.set()

    def _announce(self, session, frame):
        message = Progress(session.time, session.steps, frame)
        try:
            self.progress.put_nowait(message)
        except Full:
            try:
                self.progress.get_nowait()
            except Empty:
                pass
            self.progress.put_nowait(message)

    def _work(self, session, prefix, interval):
        error = ""
        try:
            manifest = Path(str(prefix) + ".series.json")
            formats = (
                tuple(json.loads(manifest.read_text())["formats"])
                if manifest.exists()
                else ("xdmf",)
            )
            writer = TemporalWriter(prefix, session, formats=formats)
            writer.append(session)
            frame = prefix.parent / (prefix.name + f"_{session.steps:08d}.h5")
            self._announce(session, frame)
            while not session.completed and not self.pause.is_set():
                session.step()
                if session.steps % interval == 0:
                    writer.append(session)
                    frame = prefix.parent / (
                        prefix.name + f"_{session.steps:08d}.h5"
                    )
                self._announce(session, frame)
            writer.append(session)
            frame = prefix.parent / (prefix.name + f"_{session.steps:08d}.h5")
            self._announce(session, frame)
        except Exception as exc:
            error = str(exc)
        self.finished.set()
        self.results.put(Result(session, error))
