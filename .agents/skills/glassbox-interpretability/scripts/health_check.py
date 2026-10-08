#!/usr/bin/env python3
"""
GlassBox Health Check Script
Usage: python .agents/skills/glassbox-interpretability/scripts/health_check.py

Checks:
  1. Server is running on localhost:8000
  2. /api/health returns {"model_loaded": true}
  3. WebSocket connection can be established
  4. A short generation request succeeds
"""
import sys
import json
import urllib.request
import urllib.error

SERVER = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws/generate"


def check_http(url: str, label: str) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
            print(f"  ✅ {label}: {data}")
            return data
    except urllib.error.URLError as e:
        print(f"  ❌ {label} FAILED: {e}")
        return None


def check_websocket() -> bool:
    try:
        import websocket  # pip install websocket-client
        ws = websocket.create_connection(WS_URL, timeout=5)
        ws.send(json.dumps({
            "prompt": "Say just 'OK'",
            "max_new_tokens": 5,
            "temperature": 0.0,
            "greedy": True,
        }))
        received = []
        while True:
            msg = json.loads(ws.recv())
            if msg.get("type") == "done":
                break
            if msg.get("type") == "token":
                received.append(msg["data"]["token_str"])
        ws.close()
        text = "".join(received).strip()
        print(f"  ✅ WebSocket OK — generated: {repr(text)}")
        return True
    except ImportError:
        print("  ⚠️  websocket-client not installed — skipping WS test")
        print("      Install with: pip install websocket-client")
        return True
    except Exception as e:
        print(f"  ❌ WebSocket FAILED: {e}")
        return False


def main():
    print("\n🔍 GlassBox Health Check\n")
    
    all_ok = True

    print("1. Checking HTTP server …")
    root = check_http(SERVER, "Root endpoint")
    if root is None:
        print("\n⚠️  Server is not running. Start it with:")
        print("   python run_server.py\n")
        sys.exit(1)

    print("\n2. Checking model load status …")
    health = check_http(f"{SERVER}/api/health", "Health endpoint")
    if health is None or not health.get("model_loaded"):
        print("  ⚠️  Model not yet loaded. Wait for 'Model loaded' in server logs.")
        all_ok = False

    print("\n3. Checking WebSocket …")
    ws_ok = check_websocket()
    if not ws_ok:
        all_ok = False

    print()
    if all_ok:
        print("✅ All checks passed — GlassBox is healthy!\n")
    else:
        print("❌ Some checks failed — see details above.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
