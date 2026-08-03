#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# Wait for successful SQL benchmark results for the pull request base commit.
set -Eeuo pipefail

max_attempts=120
for ((attempt = 1; attempt <= max_attempts; attempt++)); do
  runs="$(
    gh api --method GET "repos/${GITHUB_REPOSITORY}/actions/runs" \
      -f head_sha="$BASELINE_COMMIT" \
      -f event=push \
      -f per_page=100
  )"
  run_id="$(
    jq -r '
      [.workflow_runs[]
        | select(
            .path == ".github/workflows/bench.yml"
            or .path == ".github/workflows/develop-bench.yml"
          )]
      | max_by(.id)
      | .id // empty
    ' <<< "$runs"
  )"

  if [[ -n "$run_id" ]]; then
    run_status="$(jq -r --argjson run_id "$run_id" '
      .workflow_runs[]
      | select(.id == $run_id)
      | .status
    ' <<< "$runs")"

    jobs="$(
      gh api --method GET \
        "repos/${GITHUB_REPOSITORY}/actions/runs/${run_id}/jobs" \
        -f filter=latest \
        -f per_page=100
    )"
    sql_jobs="$(jq '[.jobs[] | select(.name | startswith("sql / bench ("))]' <<< "$jobs")"
    sql_job_count="$(jq length <<< "$sql_jobs")"
    failed_jobs="$(
      jq '[.[] | select(.status == "completed" and .conclusion != "success")] | length' \
        <<< "$sql_jobs"
    )"
    incomplete_jobs="$(jq '[.[] | select(.status != "completed")] | length' <<< "$sql_jobs")"

    if (( failed_jobs > 0 )); then
      echo "SQL benchmarks failed for base commit $BASELINE_COMMIT:" >&2
      jq -r '
        .[]
        | select(.status == "completed" and .conclusion != "success")
        | "  \(.name): \(.conclusion)"
      ' \
        <<< "$sql_jobs" >&2
      exit 1
    fi

    if (( sql_job_count > 0 && incomplete_jobs == 0 )); then
      echo "SQL baselines are ready for $BASELINE_COMMIT"
      exit 0
    fi

    if [[ "$run_status" == "completed" ]]; then
      echo "Develop workflow completed without SQL baselines for $BASELINE_COMMIT" >&2
      exit 1
    fi
  fi

  if (( attempt == max_attempts )); then
    echo "Timed out waiting for SQL baselines from $BASELINE_COMMIT" >&2
    exit 1
  fi

  echo "Waiting for SQL benchmarks from base commit $BASELINE_COMMIT"
  sleep 60
done
