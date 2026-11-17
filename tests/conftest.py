# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""Pytest fixtures shared across the MOSAIC test suite.

Importing ``jc_testlib`` here (first thing) forces the isolated MOSAIC_* env vars
before ``config`` is imported anywhere, so every test — unit or integration —
runs against the jc_test database / namespace / bucket, never the dev stores.
"""

import pytest

import jc_testlib


def pytest_sessionstart(session):
    """Say once per session which ingest worker binary ``mosaic ingest``
    runs (or that none is built and the tests that run ingest skip:
    ``jc_testlib.require_ingest_worker``) and which joint-call binary
    ``run_joint_call`` runs (or that none is built and the tests that run
    the joint caller skip)."""
    worker = jc_testlib.rust_worker_path()
    line = (
        f"mosaic ingest runs the Rust ingest worker {worker}"
        if worker
        else "mosaic ingest: no ingest worker binary; the tests that run "
        "ingest skip (" + jc_testlib.RUST_WORKER_SKIP_REASON + ")"
    )
    jc_worker = jc_testlib.joint_call_worker_path()
    jc_line = (
        f"joint_call.py runs the Rust joint caller {jc_worker}"
        if jc_worker
        else "joint_call.py: no joint-call binary; the tests that run the "
        "joint caller skip (" + jc_testlib.RUST_JOINT_CALL_SKIP_REASON + ")"
    )
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    for text in (line, jc_line):
        if reporter is not None:
            reporter.write_line(text)
        else:
            print(text)


@pytest.fixture(scope="session")
def docker_stack():
    """Require the local Docker stack; skip integration/e2e cleanly if absent.

    Also provisions the isolated jc_test database + jc-test-warehouse bucket once
    per session.
    """
    if not jc_testlib.stack_up():
        pytest.skip(
            "Docker stack (Postgres :5432 / Iceberg REST :8181 / MinIO :9000) "
            "not up — run `cd iceberg_test && docker-compose up -d`"
        )
    jc_testlib.ensure_test_infra()
    yield


@pytest.fixture
def clean_state(docker_stack):
    """Surgical reset before the test (no Docker bounce); yields the jc_testlib."""
    jc_testlib.reset_state()
    yield jc_testlib


@pytest.fixture
def pg_conn(docker_stack):
    """A psycopg2 connection to the isolated jc_test database."""
    conn = jc_testlib.get_pg_conn()
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture(scope="session")
def catalog(docker_stack):
    return jc_testlib.get_catalog()
