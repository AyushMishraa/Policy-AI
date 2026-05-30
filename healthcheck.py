#!/usr/bin/env python3
"""
Lightweight health-check script.

Connects to the Policy Agent TCP server, sends a ping,
expects a pong, then exits 0 (healthy) or 1 (unhealthy).

Used by Docker HEALTHCHECK and can also be run manually:
    python healthcheck.py
    echo $?   # 0 = healthy, 1 = unhealthy
"""

import asyncio
import json
import sys

from config import config

TIMEOUT = 5.0


async def check() -> bool:
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(config.SERVER_HOST, config.SERVER_PORT),
            timeout=TIMEOUT,
        )
    except Exception as exc:
        print(f"[healthcheck] Connection failed: {exc}", file=sys.stderr)
        return False

    try:
        # Consume the welcome status message first
        await asyncio.wait_for(reader.readline(), timeout=TIMEOUT)

        # Send a ping
        writer.write((json.dumps({"type": "ping"}) + "\n").encode())
        await writer.drain()

        # Expect a pong within timeout
        raw = await asyncio.wait_for(reader.readline(), timeout=TIMEOUT)
        msg = json.loads(raw.decode().strip())

        if msg.get("type") == "pong":
            print("[healthcheck] OK — server is responsive")
            return True

        print(f"[healthcheck] Unexpected response: {msg}", file=sys.stderr)
        return False

    except Exception as exc:
        print(f"[healthcheck] Error: {exc}", file=sys.stderr)
        return False
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


if __name__ == "__main__":
    ok = asyncio.run(check())
    sys.exit(0 if ok else 1)
