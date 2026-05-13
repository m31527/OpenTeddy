# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the OpenTeddy sidecar binary.

Produces a single-file executable that:
  - bundles Python + every pip dep + every .py module of OpenTeddy
  - on launch: picks a free port, prints handshake to stdout, runs uvicorn
  - lives at  desktop/src-tauri/binaries/openteddy-backend-<TARGET-TRIPLE>
    (Tauri's sidecar mechanism requires that exact naming)

Build it with:
    pyinstaller pyinstaller.spec --clean

Build it for a different arch:
    arch -arm64  pyinstaller pyinstaller.spec --clean   # Apple Silicon
    arch -x86_64 pyinstaller pyinstaller.spec --clean   # Intel
    (universal2 builds need PyInstaller 6+ and the --target-arch flag,
     scripted in desktop/scripts/build_sidecar.sh.)

Why the spec is hand-written rather than CLI-generated:
  - chromadb / pydantic-core / tokenizers all have native extensions
    PyInstaller's auto-discovery sometimes misses on first run. We
    list them explicitly via collect_all() so the bundle is stable
    across machines.
  - We need to embed the OpenTeddy module tree (orchestrator.py,
    executor.py, tools/, skills/, memory.py, etc.) without relying on
    "pyinstaller main.py" which would only see what main.py imports.
"""

from PyInstaller.utils.hooks import collect_all, collect_submodules
import os

# Project root — wherever this spec file sits.
ROOT = os.path.abspath(os.path.dirname(SPEC))   # SPEC is set by PyInstaller


# ── Heavy / extension-laden deps that need full collection ─────────────
# `collect_all(name)` returns (data_files, binaries, hidden_imports)
# for the package. Required for anything with C / Rust extensions or
# data files PyInstaller's static analyser misses.
EXTRAS_DATAS    = []
EXTRAS_BINARIES = []
EXTRAS_HIDDEN   = []

for pkg in [
    "chromadb",          # hnswlib, parquet, sqlite — needs the works
    "onnxruntime",       # ChromaDB's default embedder (ONNXMiniLM_L6_V2)
                         # lazy-imports onnxruntime when collection.add()
                         # first runs. Without it, every memory write
                         # silently raises ImportError → swallowed by
                         # add_memory's exception handler → embeddings
                         # stay at 0 forever and the Memory tab is empty
                         # despite tasks completing successfully.
    "tokenizers",        # used transitively by anthropic + chromadb
                         # default embedder (Rust ext for HF tokenisers)
    "pydantic_core",     # Rust core, easy to forget hidden imports
    "anthropic",         # tokenizers + JSON schemas
    "fastapi",           # mostly fine but pulls starlette templates
    "uvicorn",           # multiple workers, websockets
    "aiosqlite",         # SQLite native lib paths
    "httpx",             # ssl certs handling
    "httpcore",
    "h11", "h2",
    "pypdf",             # pdf_extract_text tool — pypdf is pure Python
                         # but ships ~55 submodules including _codecs/*
                         # (CID-encoding tables CJK PDFs depend on).
                         # Without collect_all the sidecar misses some
                         # submodules and extraction on Chinese /
                         # Japanese / Korean PDFs returns mojibake.
]:
    try:
        d, b, h = collect_all(pkg)
        EXTRAS_DATAS    += d
        EXTRAS_BINARIES += b
        EXTRAS_HIDDEN   += h
    except Exception as exc:  # noqa: BLE001
        print(f"[spec] collect_all({pkg}) failed: {exc} — continuing")


# ── OpenTeddy's own module tree ─────────────────────────────────────────
# Most are picked up via the entry point's import graph, but a few are
# loaded dynamically (skills/*.py via importlib, tool registry's
# auto_register_all walks the tools/ dir at runtime). Pre-collect them
# so PyInstaller doesn't strip them as "unused".
OPENTEDDY_MODULES = (
    collect_submodules("tools")
    + collect_submodules("skills")
    + [
        "orchestrator", "executor", "escalation", "skill_factory",
        "tracker", "memory", "approval_store", "settings_store",
        "tool_registry", "models", "config", "license_check",
        "model_profile",
    ]
)


# ── Bundled data dirs ────────────────────────────────────────────────────
# main.py resolves `static/` via os.path.dirname(__file__). In a
# PyInstaller-frozen binary, __file__ points inside the _MEIxxxxx temp
# dir that gets unpacked at launch. So we must explicitly copy the
# OpenTeddy data dirs there at build time, otherwise:
#   - GET / returns 500 (index.html not found, _rewrite_index_html
#     crashes on FileNotFoundError)
#   - /static/* serves nothing (StaticFiles mount skipped because
#     `os.path.isdir(_static_dir)` is False on a fresh _MEI dir)
# Format: list of (source, destination_within_bundle) tuples.
DATA_DIRS = [
    ("static", "static"),  # web UI: index.html, i18n.js, styles.css, OpenTeddy-logo.svg, …
]


# ── Analysis ─────────────────────────────────────────────────────────────
a = Analysis(
    ["sidecar_main.py"],
    pathex=[ROOT],
    binaries=EXTRAS_BINARIES,
    datas=EXTRAS_DATAS + DATA_DIRS,
    hiddenimports=EXTRAS_HIDDEN + OPENTEDDY_MODULES,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Hard-exclude things we know the backend never uses; PyInstaller
        # would otherwise add them via transitive imports and bloat the
        # bundle by tens of MB.
        "tkinter",         # GUI toolkit, never loaded
        "matplotlib",      # only used inside python_exec subprocess sandbox
        "PIL", "Pillow",   # ditto
        "pytest", "unittest",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)


# ── Single-file executable ──────────────────────────────────────────────
# `name` MUST match what Tauri expects in its `externalBin` config.
# Tauri appends the target triple (e.g. -aarch64-apple-darwin) to the
# name when copying — so the file we output here is the bare name
# `openteddy-backend`; the Tauri build pipeline renames per platform.
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="openteddy-backend",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,            # macOS codesign requires symbol-rich binary
    upx=False,              # UPX breaks codesigning on macOS
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,           # we DO want stdout/stderr — Tauri reads stdout
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,       # set by --target-arch on the CLI invocation
    codesign_identity=None,
    entitlements_file=None,
)
