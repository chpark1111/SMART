from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StageRecord:
    stage: str
    category: str
    mesh_id: str
    status: str
    started_at: float
    finished_at: float
    output_path: str | None = None
    log_path: str | None = None
    command: list[str] | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def now(
        cls,
        *,
        stage: str,
        category: str,
        mesh_id: str,
        status: str,
        started_at: float | None = None,
        output_path: str | Path | None = None,
        log_path: str | Path | None = None,
        command: list[str] | None = None,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "StageRecord":
        now = time.time()
        return cls(
            stage=stage,
            category=category,
            mesh_id=mesh_id,
            status=status,
            started_at=started_at if started_at is not None else now,
            finished_at=now,
            output_path=str(output_path) if output_path is not None else None,
            log_path=str(log_path) if log_path is not None else None,
            command=command,
            error=error,
            metadata=metadata or {},
        )


class ManifestWriter:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def append(self, record: StageRecord) -> None:
        path = self.root / f"{record.stage}.jsonl"
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(asdict(record), sort_keys=True) + "\n")

    def write_summary(self, records: list[StageRecord]) -> Path:
        summary: dict[str, dict[str, int]] = {}
        for record in records:
            stage_summary = summary.setdefault(record.stage, {})
            stage_summary[record.status] = stage_summary.get(record.status, 0) + 1
        path = self.root / "summary.json"
        path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        return path
