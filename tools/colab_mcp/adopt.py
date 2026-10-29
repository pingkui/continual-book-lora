"""Adopt a Colab runtime this host did not create (opened in the browser, or assigned after a request timed out):
register it in the CLI's session store under NAME so `colab exec/upload/download/stop -s NAME` work, and give it an
auto-stop deadline in the MCP wrapper. Run with the CLI's own interpreter:
  ~/.local/share/uv/tools/google-colab-cli/bin/python adopt.py NAME ENDPOINT MAX_MINUTES
ENDPOINT comes from the account's assignment list (the wrapper's sessions tool, or list_assignments)."""
import datetime
import json
import sys
import time
from pathlib import Path

from colab_cli.common import state
from colab_cli.state import SessionState

name, endpoint, minutes = sys.argv[1], sys.argv[2], int(sys.argv[3])
if state.store.get(name):
    sys.exit(f"a session named {name!r} already exists")
match = [a for a in state.client.list_assignments() if a.endpoint == endpoint]
if not match:
    sys.exit(f"no live assignment {endpoint!r}")
a = match[0]
info = a.runtime_proxy_info
acc = a.accelerator.value if hasattr(a.accelerator, "value") else str(a.accelerator)
variant = "TPU" if acc.upper().startswith("V") else ("DEFAULT" if acc.upper() == "NONE" else "GPU")
state.store.add(SessionState(
    name=name, token=info.token, url=info.url, endpoint=a.endpoint,
    token_expires_at=info.expires_at() if hasattr(info, "expires_at") else None,
    variant=variant, accelerator=acc,
    machine_shape=a.machine_shape.name if hasattr(a.machine_shape, "name") else "STANDARD"))

wrapper = Path.home() / ".config/colab-mcp-wrapper"
now = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
for fname, update in (("deadlines.json", {name: time.time() + 60 * minutes}),
                      ("open.json", {name: {"acc": acc.lower(), "start": now, "t0": time.time()}})):
    p = wrapper / fname
    d = json.loads(p.read_text() or "{}") if p.exists() else {}
    d.update(update)
    p.write_text(json.dumps(d))
with open(wrapper / "usage.jsonl", "a") as f:
    f.write(json.dumps({"t": now, "event": "open", "name": name, "acc": acc.lower(), "source": f"adopted {endpoint}"}) + "\n")
(Path.home() / ".config/colab-cli/colab.log").unlink(missing_ok=True)
print(f"adopted {endpoint} as {name!r} ({acc}); auto-stop in {minutes} min")
