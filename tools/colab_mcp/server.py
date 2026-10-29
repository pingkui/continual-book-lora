"""Colab MCP server: a thin wrapper around the official google-colab-cli (`colab`), for a headless remote host.

Login uses the CLI's copy-paste OAuth flow (login_start -> user opens the URL -> login_finish with the code).
Files registered with colab_launch(fetch=...) and its log are downloaded to RESULTS/<session>-<time>/ before the
session is stopped, by colab_stop or by the deadline. Safety net against forgotten sessions: every session gets a deadline (max_minutes, default 60); a background
thread stops it when the deadline passes. Deadlines are kept in a file, so a restarted server still enforces them
on its next start or tool call.
Usage log (STATE/usage.jsonl, one JSON object per line, UTC): every session opened or refused, stopped (by hand or by
the deadline) or lost (ended by Colab), with its accelerator and minutes; usage_log() summarises it. Colab does not
publish its free-tier limits, so this is how they get measured.
"""
import datetime
import re
import json
import os
import shlex
import subprocess
import threading
import time
from pathlib import Path

from mcp.server.mcpserver import MCPServer

COLAB = os.environ.get("COLAB_BIN", str(Path.home() / ".local/bin/colab"))
STATE = Path.home() / ".config/colab-mcp-wrapper"
STATE.mkdir(parents=True, exist_ok=True)
# The CLI keeps its token and session list under ~/.config/colab-cli, shared with other Claude sessions on this host.
CLI_ENV = None  # inherit the environment
CLI_DIR = Path.home() / ".config/colab-cli"
DEADLINES = STATE / "deadlines.json"
FIFO = STATE / "login.in"
LOGIN_OUT = STATE / "login.out"
CLI_LOG = CLI_DIR / "colab.log"
FETCH = STATE / "fetch.json"  # session -> remote files to download before the session is stopped
RESULTS = Path(os.environ.get("COLAB_RESULTS", "/home/ubuntu/kaggle/tpu/results"))
USAGE = STATE / "usage.jsonl"
OPEN = STATE / "open.json"  # session -> {"acc", "start"} for sessions this wrapper opened and has not seen end
_lock = threading.Lock()

mcp = MCPServer("colab", instructions=(
    "Google Colab via the official colab CLI. Free-tier TPU/GPU time is limited and not published: tell the user the "
    "expected duration before colab_new with an accelerator, keep max_minutes tight, and call colab_stop as soon as a "
    "job is done. Long jobs: colab_launch (background on the VM) + colab_tail, not colab_exec."))


def _run(args, timeout=120, stdin=None):
    try:
        # never let the CLI inherit our stdin: that is the MCP protocol pipe
        io = {"input": stdin} if stdin is not None else {"stdin": subprocess.DEVNULL}
        p = subprocess.run([COLAB, *args], capture_output=True, text=True, timeout=timeout, env=CLI_ENV, **io)
        CLI_LOG.unlink(missing_ok=True)  # the CLI logs token responses there
        out = (p.stdout + p.stderr).strip()
        for m in re.finditer(r"Session '([^']+)' (?:appears to be lost|not found)", out):
            _ended(m.group(1), "lost")
        if "To authorize colab-cli" in out:
            return "Not logged in: call login_start, then login_finish with the code."
        return out[-6000:] if out else f"(exit {p.returncode}, no output)"
    except subprocess.TimeoutExpired:
        return f"timed out after {timeout}s"


CLI_PY = str(Path(COLAB).resolve().parent / "python")  # the CLI's own interpreter (uv tool venv)


def _assign_api(code, timeout=90):
    """Run code against the CLI's client (state.client) in its own interpreter; returns stdout ('' on failure)."""
    try:
        p = subprocess.run([CLI_PY, "-c", "from colab_cli.common import state\n" + code], capture_output=True,
                           text=True, timeout=timeout, stdin=subprocess.DEVNULL, env=CLI_ENV)
        CLI_LOG.unlink(missing_ok=True)
        return p.stdout
    except (subprocess.TimeoutExpired, OSError):
        return ""


def _endpoints():
    out = _assign_api("print('\\n'.join(a.endpoint for a in state.client.list_assignments()))")
    return set(out.split())


def _release_late(before, name, acc, wait=150):
    """A request the CLI gave up on can still be assigned on the server a little later, and then it runs (and is
    billed) with no session entry, so nothing would stop it. Watch for endpoints that were not there before this
    request and release them; endpoints that already existed are never touched."""
    t0 = time.time()
    while time.time() - t0 < wait:
        new = _endpoints() - before
        if new:
            for e in new:
                _assign_api(f"state.client.unassign({e!r})")
                _log("orphan_released", name=name, acc=acc, endpoint=e, after_s=round(time.time() - t0))
            return f"\nreleased {len(new)} runtime(s) assigned after the timeout: {sorted(new)}"
        time.sleep(15)
    return ""


def _deadlines():
    return _load(DEADLINES)


def _save(d):
    DEADLINES.write_text(json.dumps(d))


def _load(f):
    try:
        return json.loads(f.read_text())
    except (OSError, ValueError):
        return {}


def _now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log(event, **kw):
    with open(USAGE, "a") as f:
        f.write(json.dumps({"t": _now(), "event": event, **kw}) + "\n")


def _ended(name, how):
    """Record the end of a session this wrapper opened (how: stop / deadline / lost / gone)."""
    o = _load(OPEN)
    if name not in o:
        return
    rec = o.pop(name)
    OPEN.write_text(json.dumps(o))
    _log(how, name=name, acc=rec["acc"], start=rec["start"], minutes=round((time.time() - rec["t0"]) / 60, 1))


def _fetch(name):
    """Download the files registered for `name`; return a one-line report."""
    paths = _load(FETCH).get(name, [])
    if not paths:
        return ""
    dest = RESULTS / f"{name}-{time.strftime('%Y%m%d-%H%M%S')}"
    dest.mkdir(parents=True, exist_ok=True)
    got, miss = [], []
    for rp in paths:
        _run(["download", "-s", name, rp, str(dest / Path(rp).name)], timeout=300)
        (got if (dest / Path(rp).name).exists() else miss).append(Path(rp).name)
    d = _load(FETCH)
    d.pop(name, None)
    FETCH.write_text(json.dumps(d))
    return f"downloaded to {dest}: {got}" + (f"; missing: {miss}" if miss else "")


def _reap():
    """Stop every session whose deadline has passed; return what was stopped."""
    with _lock:
        d, now, stopped = _deadlines(), time.time(), []
        for name, t in list(d.items()):
            if now >= t:
                _fetch(name)
                _run(["stop", "-s", name], timeout=90)
                _ended(name, "deadline")
                stopped.append(name)
                del d[name]
        _save(d)
    return stopped


def _watchdog():
    while True:
        _reap()
        time.sleep(60)


threading.Thread(target=_watchdog, daemon=True).start()


@mcp.tool()
def login_start() -> str:
    """Begin the OAuth copy-paste login. Returns a URL for the user to open; then call login_finish with the code."""
    for f in (FIFO, LOGIN_OUT):
        f.unlink(missing_ok=True)
    os.mkfifo(FIFO)
    subprocess.Popen(["bash", "-c", f"exec 3<>{shlex.quote(str(FIFO))}; {shlex.quote(COLAB)} sessions <&3 > "
                      f"{shlex.quote(str(LOGIN_OUT))} 2>&1"], start_new_session=True, stdin=subprocess.DEVNULL,
                     env=CLI_ENV)
    for _ in range(30):
        time.sleep(0.5)
        txt = LOGIN_OUT.read_text() if LOGIN_OUT.exists() else ""
        for line in txt.splitlines():
            if "https://accounts.google.com" in line:
                return line.strip().replace("&amp;", "&")
        if "No active sessions" in txt or "Session" in txt:
            return "Already logged in: " + txt.strip()[-300:]
    return "No login URL appeared: " + (LOGIN_OUT.read_text()[-500:] if LOGIN_OUT.exists() else "")


@mcp.tool()
def login_finish(code: str) -> str:
    """Pass the authorization code shown by Google after approving login_start's URL. Returns the logged-in email."""
    if not FIFO.exists():
        return "No login in progress; call login_start first."
    with open(FIFO, "w") as f:
        f.write(code.strip() + "\n")
    time.sleep(8)
    out = LOGIN_OUT.read_text() if LOGIN_OUT.exists() else ""
    FIFO.unlink(missing_ok=True)
    CLI_LOG.unlink(missing_ok=True)
    tok = CLI_DIR / "token.json"
    if tok.exists():
        tok.chmod(0o600)
        return "Logged in. " + whoami()
    return "Login failed: " + "\n".join(l for l in out.splitlines() if "accounts.google" not in l)[-800:]


@mcp.tool()
def whoami() -> str:
    """Email and scopes of the logged-in account."""
    out = _run(["whoami"], timeout=30)
    return "\n".join(l for l in out.splitlines() if "token" not in l.lower())


@mcp.tool()
def usage() -> str:
    """Compute-unit balance, current usage rate per hour and active assignments."""
    return _run(["usage"], timeout=60)


@mcp.tool()
def sessions() -> str:
    """Sessions on the server, plus the auto-stop deadline this wrapper holds for each."""
    stopped = _reap()
    left = {n: f"{(t - time.time()) / 60:.0f} min" for n, t in _deadlines().items()}
    out = _run(["sessions"], timeout=60)
    if "Not logged in" not in out and "timed out" not in out:
        live = set(re.findall(r"^\[([^\]]+)\]", out, re.M))
        for name in list(_load(OPEN)):
            if name not in live:  # ended without us seeing it; the end time is only an upper bound
                _ended(name, "gone")
    return out + f"\nauto-stop in: {left}" + (f"\nauto-stopped now: {stopped}" if stopped else "")


@mcp.tool()
def colab_new(name: str, accelerator: str = "cpu", max_minutes: int = 60, high_mem: bool = False) -> str:
    """Create a session. accelerator: 'cpu', a GPU (T4, L4, G4, H100, A100) or a TPU (v5e1, v6e1).
    high_mem: high-RAM machine shape (CPU, T4, A100; needs Colab Pro/Pro+; ignored for L4 and TPUs).
    The session is stopped automatically after max_minutes (1-720)."""
    max_minutes = max(1, min(int(max_minutes), 720))
    acc = accelerator.strip()
    args = ["new", "-s", name] + (["--high-mem"] if high_mem else [])
    if acc.lower().startswith("v"):
        args += ["--tpu", acc.lower()]
    elif acc.lower() != "cpu":
        args += ["--gpu", acc.upper()]
    tag = acc.lower() + ("-hm" if high_mem else "")  # usage-log label
    before = _endpoints()
    out = _run(args, timeout=600)  # a busy TPU pool can take several minutes to assign
    if "READY" in out:
        with _lock:
            d = _deadlines()
            d[name] = time.time() + 60 * max_minutes
            _save(d)
            o = _load(OPEN)
            o[name] = {"acc": tag, "start": _now(), "t0": time.time()}
            OPEN.write_text(json.dumps(o))
        _log("open", name=name, acc=tag)
        out += f"\nauto-stop in {max_minutes} min"
    else:
        # last matching line outside the traceback box (whose lines start with box-drawing characters)
        plain = [l.strip() for l in out.splitlines() if l.strip() and l.strip()[0] not in "│╭╰"]
        why = next((l for l in reversed(plain) if re.search(r"503|Unavailable|quota|limit|Error", l, re.I)),
                   plain[-1] if plain else "")
        _log("refused", name=name, acc=tag, reason=why[:200])
        if out.startswith("timed out") or "Read timed out" in out or "ReadTimeout" in out:
            out += _release_late(before, name, tag)  # the server may still assign the request later
    return out


@mcp.tool()
def colab_stop(name: str) -> str:
    """Download the files registered by colab_launch(fetch=...), then stop the session and release its VM."""
    with _lock:
        d = _deadlines()
        d.pop(name, None)
        _save(d)
    got = _fetch(name)
    out = _run(["stop", "-s", name], timeout=90)
    _ended(name, "stop")
    return (got + "\n" if got else "") + out


@mcp.tool()
def colab_extend(name: str, minutes: int) -> str:
    """Move a session's auto-stop deadline to `minutes` from now (1-720)."""
    with _lock:
        d = _deadlines()
        if name not in d:
            return f"No deadline held for {name}."
        d[name] = time.time() + 60 * max(1, min(int(minutes), 720))
        _save(d)
    return f"{name}: auto-stop in {minutes} min"


@mcp.tool()
def colab_upload(name: str, local_path: str, remote_path: str) -> str:
    """Upload a local file to the session VM (e.g. /content/data.tgz)."""
    return _run(["upload", "-s", name, local_path, remote_path], timeout=600)


@mcp.tool()
def colab_download(name: str, remote_path: str, local_path: str = "") -> str:
    """Download one file from the session VM (default: into RESULTS/<session>/)."""
    lp = Path(local_path) if local_path else RESULTS / name / Path(remote_path).name
    lp.parent.mkdir(parents=True, exist_ok=True)
    out = _run(["download", "-s", name, remote_path, str(lp)], timeout=300)
    return out + (f"\nsaved {lp} ({lp.stat().st_size} bytes)" if lp.exists() else "\nnot saved")


@mcp.tool()
def colab_exec(name: str, code: str = "", file: str = "", timeout_s: int = 300) -> str:
    """Run Python in the session's kernel, from `code` or a local .py/.ipynb `file`, and return its output.
    Output arrives only when it finishes; for anything long use colab_launch. Note: code that imports a TPU
    runtime in the kernel keeps the TPU busy until colab_restart_kernel."""
    if file:
        return _run(["exec", "-s", name, "-f", file], timeout=timeout_s)
    return _run(["exec", "-s", name], timeout=timeout_s, stdin=code)


@mcp.tool()
def colab_launch(name: str, command: str, log: str = "/content/job.log", fetch: str = "") -> str:
    """Start a shell command in the background on the VM (e.g. 'cd /content && python train.py'), output to `log`.
    `fetch`: comma-separated remote result files; they and the log are downloaded automatically before the session
    is stopped (colab_stop or deadline). Returns immediately; read progress with colab_tail."""
    with _lock:
        d = _load(FETCH)
        d[name] = list(dict.fromkeys(d.get(name, []) + [log] + [f.strip() for f in fetch.split(",") if f.strip()]))
        FETCH.write_text(json.dumps(d))
    code = ("import subprocess, os\n"
            f"subprocess.Popen({'(' + command + ') > ' + log + ' 2>&1'!r}, shell=True, env=dict(os.environ, PYTHONUNBUFFERED='1'))\n"
            "print('launched')")
    return _run(["exec", "-s", name], timeout=120, stdin=code)


@mcp.tool()
def colab_tail(name: str, log: str = "/content/job.log", lines: int = 40, grep: str = "") -> str:
    """Last `lines` lines of a log on the VM, optionally only lines containing `grep`."""
    code = (f"s = open({log!r}, errors='replace').read().replace('\\r', '\\n').splitlines()\n"
            f"s = [l for l in s if {grep!r} in l]\n"
            f"print(len(s), 'lines'); print('\\n'.join(s[-{int(lines)}:])[-6000:])")
    return _run(["exec", "-s", name], timeout=90, stdin=code)


@mcp.tool()
def usage_log(days: int = 7) -> str:
    """Accelerator time per UTC day from the usage log, every refusal with the accelerator minutes used in the 3 and
    24 hours before it, and sessions still open. For estimating Colab's unpublished free-tier limits."""
    cut = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=int(days))).strftime("%Y-%m-%dT%H:%M:%SZ")
    ev = []
    if USAGE.exists():
        for l in USAGE.read_text().splitlines():
            try:
                e = json.loads(l)
            except ValueError:
                continue
            if e["t"] >= cut:
                ev.append(e)
    ends = [e for e in ev if e["event"] in ("stop", "deadline", "lost", "gone")]
    per_day = {}
    for e in ends:
        k = (e["start"][:10], e["acc"])
        per_day[k] = per_day.get(k, 0) + e["minutes"]
    ts = lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%SZ")

    def used_before(t, hours):  # minutes of sessions overlapping (t - hours, t)
        lo, tot = ts(t) - datetime.timedelta(hours=hours), 0.0
        for e in ends:
            a, b = ts(e["start"]), ts(e["start"]) + datetime.timedelta(minutes=e["minutes"])
            tot += max(0.0, (min(b, ts(t)) - max(a, lo)).total_seconds() / 60)
        return round(tot)

    lines = ["minutes per UTC day (day of session start): " + ", ".join(f"{d} {a}: {m:.0f}" for (d, a), m in sorted(per_day.items()))]
    lines += [f"{e['event']:8s} {e['start']} -> +{e['minutes']} min  {e['name']} ({e['acc']})" + (" [end time is an upper bound]" if e["event"] == "gone" or e.get("upper_bound") else "")
              for e in ends]
    for e in ev:
        if e["event"] == "refused":
            lines.append(f"refused  {e['t']} {e['acc']}: used {used_before(e['t'], 3)} min in the 3 h before, "
                         f"{used_before(e['t'], 24)} min in the 24 h before; {e.get('reason', '')[:80]}")
    for n, o in _load(OPEN).items():
        lines.append(f"open     {o['start']} {n} ({o['acc']}), {(time.time() - o['t0']) / 60:.0f} min so far")
    return "\n".join(lines) if ev or _load(OPEN) else "usage log is empty"


@mcp.tool()
def colab_restart_kernel(name: str) -> str:
    """Restart the session's Jupyter kernel (releases a TPU held by code run in the kernel)."""
    return _run(["restart-kernel", "-s", name], timeout=90)


@mcp.tool()
def logout() -> str:
    """Revoke the Google OAuth grant and delete the local token and CLI state."""
    tok = CLI_DIR / "token.json"
    msg = "no token"
    if tok.exists():
        import urllib.parse
        import urllib.request
        t = json.loads(tok.read_text())
        val = t.get("refresh_token") or t.get("token") or t.get("access_token")
        try:
            req = urllib.request.Request("https://oauth2.googleapis.com/revoke",
                                         data=urllib.parse.urlencode({"token": val}).encode(),
                                         headers={"Content-Type": "application/x-www-form-urlencoded"})
            msg = f"revoke status {urllib.request.urlopen(req, timeout=30).status}"
        except Exception as e:  # still delete locally
            msg = f"revoke failed: {e!r}"[:200]
    subprocess.run(["rm", "-rf", str(CLI_DIR)])
    return msg + "; local Colab CLI state deleted"


if __name__ == "__main__":
    _reap()
    mcp.run("stdio")
