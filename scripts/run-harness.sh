#!/usr/bin/env bash
# Usage: ./scripts/run-harness.sh <scenario.toml> <out-dir>
# Runs the Bevy client through a scripted scenario. Prints absolute
# PNG paths on stdout (one per line). All Bevy logs go to stderr.

set -euo pipefail

if [[ "${1:-}" == "" || "${2:-}" == "" ]]; then
    echo "Usage: $0 <scenario.toml> <out-dir>" >&2
    exit 2
fi

SCENARIO="$1"
OUTDIR="$2"
mkdir -p "$OUTDIR"
OUTDIR_ABS="$(cd "$OUTDIR" && pwd)"

# Build release: debug streaming is too slow.
echo ">>> cargo build --release -p atomr-worlds-client" >&2
cargo build --release -p atomr-worlds-client >&2

CLIENT="./target/release/atomr-worlds-client"

if command -v xvfb-run >/dev/null 2>&1; then
    echo ">>> running under xvfb-run (software GL)" >&2
    export WGPU_BACKEND=gl
    export LIBGL_ALWAYS_SOFTWARE=1
    export RUST_LOG="${RUST_LOG:-warn}"
    xvfb-run -a -s "-screen 0 1920x1080x24" \
        "$CLIENT" --harness "$SCENARIO" --harness-out "$OUTDIR_ABS"
else
    echo ">>> xvfb-run not installed; using current DISPLAY=${DISPLAY:-<unset>}" >&2
    if [[ -z "${DISPLAY:-}" ]]; then
        echo "ERROR: no DISPLAY set and xvfb-run not installed. Install with: sudo apt-get install xvfb" >&2
        exit 3
    fi
    export RUST_LOG="${RUST_LOG:-warn}"
    "$CLIENT" --harness "$SCENARIO" --harness-out "$OUTDIR_ABS"
fi
