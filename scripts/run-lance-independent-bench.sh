#!/usr/bin/env bash

set -Eeuo pipefail

phase=${1:-all}
repo_root=$(git rev-parse --show-toplevel)
results_dir=${repo_root}/independent-results
mkdir -p "${results_dir}"

export PATH="${HOME}/.cargo/bin:${HOME}/.local/bin:${PATH}"
export RUSTFLAGS="-C target-cpu=native -C force-frame-pointers=yes"
export RUST_BACKTRACE=full
export VORTEX_EXPERIMENTAL_PATCHED_ARRAY=1
export FLAT_LAYOUT_INLINE_ARRAY_NODE=1

run_timed() {
    local name=$1
    shift
    /usr/bin/time -v -o "${results_dir}/${name}.resources.txt" "$@"
}

capture_system() {
    {
        git rev-parse HEAD
        rustc -Vv
        uname -a
        lscpu
        numactl --hardware
        lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINTS,MODEL
        df -h
    } >"${results_dir}/system.txt"
}

build_benchmarks() {
    capture_system
    cargo build --locked --package compress-bench \
        --profile release_debug --features lance,unstable_encodings
    cargo build --locked --package random-access-bench \
        --profile release_debug --features lance,unstable_encodings
    cargo build --locked --bin data-gen --bin datafusion-bench --bin lance-bench \
        --profile release_debug --features unstable_encodings
    uv sync --project bench-orchestrator
}

run_compression() {
    local version version_slug
    for version in 2.0 2.1 2.3; do
        version_slug=${version/./_}
        LANCE_FILE_VERSION=${version} run_timed "compression-v${version_slug}" \
            bash scripts/bench-taskset.sh target/release_debug/compress-bench \
            --formats parquet,lance,vortex --iterations 5 \
            --display-format gh-json \
            --output-path "${results_dir}/compression-v${version_slug}.json" \
            --ingest-jsonl "${results_dir}/compression-v${version_slug}.ingest.jsonl"
    done
}

run_random_access() {
    rm -rf parts results.json results.ingest.jsonl
    run_timed random-access python3 scripts/random-access-split.py --emit-ingest-records
    mv results.json "${results_dir}/random-access.json"
    mv results.ingest.jsonl "${results_dir}/random-access.ingest.jsonl"
}

run_sql() {
    local benchmark=$1
    local scale_factor=$2
    local iterations=$3
    local slug=$4
    local scale_args=()
    if [[ -n ${scale_factor} ]]; then
        scale_args=(--opt "scale-factor=${scale_factor}")
    fi

    uv run --project bench-orchestrator vx-bench prepare-data "${benchmark}" \
        --formats-json '["parquet","vortex"]' "${scale_args[@]}"
    run_timed "${slug}" bash scripts/bench-taskset.sh \
        uv run --project bench-orchestrator vx-bench run "${benchmark}" \
        --targets-json '[{"engine":"datafusion","format":"parquet"},{"engine":"datafusion","format":"vortex"},{"engine":"datafusion","format":"lance"}]' \
        --iterations "${iterations}" \
        --output "${results_dir}/${slug}.json" \
        --ingest-jsonl "${results_dir}/${slug}.ingest.jsonl" \
        --no-build "${scale_args[@]}"
}

case "${phase}" in
    build)
        build_benchmarks
        ;;
    compression)
        run_compression
        ;;
    random-access)
        run_random_access
        ;;
    clickbench)
        run_sql clickbench "" 5 clickbench
        ;;
    clickbench-sorted)
        run_sql clickbench-sorted "" 5 clickbench-sorted
        ;;
    tpch-sf1)
        run_sql tpch 1.0 10 tpch-sf1
        ;;
    tpch-sf10)
        run_sql tpch 10.0 10 tpch-sf10
        ;;
    all)
        build_benchmarks
        run_compression
        run_random_access
        run_sql clickbench "" 5 clickbench
        run_sql clickbench-sorted "" 5 clickbench-sorted
        run_sql tpch 1.0 10 tpch-sf1
        run_sql tpch 10.0 10 tpch-sf10
        ;;
    *)
        echo "Unknown phase '${phase}'." >&2
        exit 2
        ;;
esac
