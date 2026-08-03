# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import json
import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WAIT_SCRIPT = REPO_ROOT / "scripts" / "wait-for-sql-baseline.sh"


def run_wait_script(
    tmp_path: Path,
    run_status: str,
    jobs: list[dict[str, str | None]],
) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    gh = bin_dir / "gh"
    gh.write_text(
        """#!/usr/bin/env bash
if [[ "$*" == *"/jobs"* ]]; then
  printf '%s\n' "$MOCK_JOBS"
else
  printf '{"workflow_runs":[{"id":42,"path":".github/workflows/develop-bench.yml","status":"%s"}]}\n' \
    "$MOCK_RUN_STATUS"
fi
""",
        encoding="utf-8",
    )
    gh.chmod(0o755)

    sleep = bin_dir / "sleep"
    sleep.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    sleep.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "BASELINE_COMMIT": "base-commit",
            "GITHUB_REPOSITORY": "vortex-data/vortex",
            "MOCK_JOBS": json.dumps({"jobs": jobs}),
            "MOCK_RUN_STATUS": run_status,
            "PATH": f"{bin_dir}:{env['PATH']}",
        }
    )
    return subprocess.run(
        ["bash", str(WAIT_SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def sql_job(name: str, status: str, conclusion: str | None) -> dict[str, str | None]:
    return {
        "name": f"sql / bench ({name})",
        "status": status,
        "conclusion": conclusion,
    }


def test_succeeds_when_sql_jobs_finish_before_workflow(tmp_path: Path) -> None:
    result = run_wait_script(
        tmp_path,
        "in_progress",
        [
            sql_job("tpch", "completed", "success"),
            sql_job("clickbench", "completed", "success"),
        ],
    )

    assert result.returncode == 0
    assert "SQL baselines are ready for base-commit" in result.stdout


def test_waits_while_any_sql_job_is_incomplete(tmp_path: Path) -> None:
    result = run_wait_script(
        tmp_path,
        "in_progress",
        [
            sql_job("tpch", "completed", "success"),
            sql_job("clickbench", "in_progress", None),
        ],
    )

    assert result.returncode == 1
    assert "Timed out waiting for SQL baselines from base-commit" in result.stderr
    assert "SQL benchmarks failed" not in result.stderr


def test_fails_immediately_when_sql_job_fails(tmp_path: Path) -> None:
    result = run_wait_script(
        tmp_path,
        "in_progress",
        [
            sql_job("tpch", "completed", "failure"),
            sql_job("clickbench", "in_progress", None),
        ],
    )

    assert result.returncode == 1
    assert "SQL benchmarks failed for base commit base-commit" in result.stderr
    assert "sql / bench (tpch): failure" in result.stderr
    assert "clickbench" not in result.stderr
