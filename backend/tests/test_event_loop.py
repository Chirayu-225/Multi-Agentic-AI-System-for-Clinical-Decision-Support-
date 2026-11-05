"""Blocking work (NER, vector store, PDF scan) must not run on the event loop.

While it does, every other request (including /health) and the WebSocket progress stream stall
for the whole duration of an upload.
"""
import asyncio
import threading
import time

import pytest

import main
from src.tools import tool_registry
from tests.test_api_flow import CLIN, client, fresh_state, login, upload  # noqa: F401


def _blocking(record):
    """A stand-in for slow synchronous work that notes which thread ran it."""
    def fn(*args, **kwargs):
        record["thread"] = threading.current_thread()
        time.sleep(0.3)
        return {} if not record.get("as_list") else []
    return fn


@pytest.mark.parametrize("name, args, as_list", [
    ("extract_entities", ("text",), False),
    ("detect_trends", ("P1", []), False),
    ("get_patient_history", ("P1",), True),
    ("get_similar_history", ("P1", "query"), False),
])
def test_blocking_tools_run_in_worker_threads_and_leave_the_loop_free(monkeypatch, name, args, as_list):
    record = {"as_list": as_list}
    monkeypatch.setattr(tool_registry, f"_{name}_sync", _blocking(record))

    async def scenario():
        ticks = 0
        done = asyncio.Event()

        async def ticker():
            nonlocal ticks
            while not done.is_set():
                ticks += 1
                await asyncio.sleep(0.02)

        t = asyncio.create_task(ticker())
        await getattr(tool_registry, name)(*args)
        done.set()
        await t
        return ticks

    ticks = asyncio.run(scenario())
    assert record["thread"] is not threading.main_thread()
    # If the 0.3 s of work had blocked the loop, the ticker would have run ~0-1 times.
    assert ticks >= 5, f"event loop was blocked (only {ticks} ticks)"


def test_upload_stores_history_and_scans_pdf_off_the_event_loop(client, monkeypatch):
    seen = {}

    def in_event_loop() -> bool:
        try:
            asyncio.get_running_loop()
            return True
        except RuntimeError:
            return False

    def fake_store(**kwargs):
        seen["store_on_loop"] = in_event_loop()

    def fake_scan(path):
        seen["scan_on_loop"] = in_event_loop()
        return []

    monkeypatch.setattr(main, "chroma_store", fake_store)
    monkeypatch.setattr(main, "detect_hidden_pdf_text", fake_scan)
    assert upload(client, login(client, CLIN)).status_code == 200
    assert seen == {"store_on_loop": False, "scan_on_loop": False}
