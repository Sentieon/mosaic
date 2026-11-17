#!/usr/bin/env bash
# Copyright (c) 2026 Sentieon Inc. All rights reserved
#
# Single entry point for the MOSAIC test suite.
#
#   ./run_tests.sh --fast          # unit tests only (no Docker) — the commit gate
#   ./run_tests.sh --integration   # integration tests (needs the Docker stack)
#   ./run_tests.sh --e2e           # full pipeline vs vcfdiff baseline (Docker, slow)
#   ./run_tests.sh --all           # unit -> restart -> other integration -> e2e
#   ./run_tests.sh --fresh-stack … # bounce the Docker stack first (destructive)
#
# Tests isolate themselves to a dedicated jc_test database / jc_test Iceberg
# namespace / jc-test-warehouse bucket (forced in tests/conftest.py), so they never
# touch the developer's MOSAIC pipeline data. The integration/e2e tiers
# require the local stack: cd iceberg_test && docker-compose up -d
set -uo pipefail

cd "$(dirname "$0")"

# Interpreter with the full stack + pytest installed. Defaults to `python3`;
# point MOSAIC_TEST_PYTHON at the right venv/conda env if that isn't it.
PYTHON="${MOSAIC_TEST_PYTHON:-python3}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MOSAIC_S3_ENDPOINT="${MOSAIC_S3_ENDPOINT:-http://localhost:9000}"

MODE="--fast"
FRESH_STACK=0
PYTEST_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --fast|--integration|--e2e|--all) MODE="$arg" ;;
        --fresh-stack) FRESH_STACK=1 ;;
        *) PYTEST_ARGS+=("$arg") ;;
    esac
done

run() {
    echo ">>> pytest $*"
    # ${arr[@]+...} guards against the empty-array "unbound variable" error under
    # `set -u` on macOS's bash 3.2.
    "$PYTHON" -m pytest "$@" ${PYTEST_ARGS[@]+"${PYTEST_ARGS[@]}"}
}

stack_up() {
    "$PYTHON" - <<'PY'
import socket, sys
def up(p):
    try:
        socket.create_connection(("localhost", p), 1).close(); return True
    except OSError:
        return False
sys.exit(0 if (up(5432) and up(8181) and up(9000)) else 1)
PY
}

if [[ "$FRESH_STACK" == 1 ]]; then
    echo ">>> --fresh-stack: bouncing the Docker stack (destructive) via mosaic clean"
    "$PYTHON" -m mosaic.cli clean
fi

need_stack() {
    if ! stack_up; then
        echo "ERROR: Docker stack (Postgres :5432 / Iceberg REST :8181 / MinIO :9000) is not up." >&2
        echo "       Start it with: cd iceberg_test && docker-compose up -d" >&2
        exit 2
    fi
}

rc=0
case "$MODE" in
    --fast)
        run -m unit || rc=$?
        ;;
    --integration)
        need_stack
        run -m integration || rc=$?
        ;;
    --e2e)
        need_stack
        run -m e2e || rc=$?
        ;;
    --all)
        run -m unit || rc=$?
        need_stack
        # Restart correctness first (the priority), then the rest, then e2e.
        run -m restart || rc=$?
        run -m "integration and not restart" || rc=$?
        run -m e2e || rc=$?
        ;;
esac

echo
if [[ "$rc" == 0 ]]; then echo "ALL SELECTED TESTS PASSED"; else echo "TESTS FAILED (rc=$rc)"; fi
exit "$rc"
