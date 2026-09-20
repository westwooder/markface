# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec: self-contained onedir build, no Python needed on the target."""
import shutil
from pathlib import Path

ROOT = Path(SPECPATH)

# --- bundle ffmpeg under a stable name so video.ffmpeg_exe() finds it ---
datas = []
try:
    import imageio_ffmpeg
    src = Path(imageio_ffmpeg.get_ffmpeg_exe())
    staged = ROOT / "build" / "ffmpeg" / "ffmpeg.exe"
    staged.parent.mkdir(parents=True, exist_ok=True)
    if not staged.exists() or staged.stat().st_size != src.stat().st_size:
        shutil.copy2(src, staged)
    datas.append((str(staged), "ffmpeg"))
except Exception as exc:
    print(f"WARNING: ffmpeg not bundled ({exc}); the app will look for it on PATH")

# --- models ---
models = ROOT / "models"
for p in sorted(models.glob("*.onnx")):
    datas.append((str(p), "models"))
datas.append((str(models / "manifest.json"), "models"))

hiddenimports = [
    "onnxruntime",
    "onnxruntime.capi._pybind_state",
]

# Qt modules we never touch; dropping them saves ~120MB.
excludes = [
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngine",
    "PySide6.QtQuick", "PySide6.QtQuick3D", "PySide6.QtQml", "PySide6.Qt3DCore",
    "PySide6.QtCharts", "PySide6.QtDataVisualization", "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets", "PySide6.QtPdf", "PySide6.QtPdfWidgets",
    "PySide6.QtBluetooth", "PySide6.QtNfc", "PySide6.QtPositioning",
    "PySide6.QtSql", "PySide6.QtTest", "PySide6.QtDesigner", "PySide6.QtHelp",
    "PySide6.QtSerialPort", "PySide6.QtSensors", "PySide6.QtSpatialAudio",
    "PySide6.QtNetworkAuth", "PySide6.QtRemoteObjects", "PySide6.QtScxml",
    "PySide6.QtSvgWidgets", "PySide6.QtTextToSpeech", "PySide6.QtUiTools",
    "PySide6.QtWebChannel", "PySide6.QtWebSockets",
    "tkinter", "unittest", "pydoc", "doctest", "pdb",
    "matplotlib", "scipy", "pandas", "PIL", "torch", "torchvision",
    "setuptools", "pip", "sympy", "onnx",
]

def _drop_duplicate_ffmpeg(items):
    """imageio_ffmpeg ships its own 84MB copy of ffmpeg; we already staged one
    under ffmpeg/. Keep a single copy - the app never calls imageio_ffmpeg's
    binary path at runtime in a frozen build."""
    out = []
    for entry in items:
        dest = entry[0].replace("\\", "/")
        if "imageio_ffmpeg/binaries/" in dest and dest.endswith(".exe"):
            print(f"spec: dropping duplicate {dest}")
            continue
        out.append(entry)
    return out


a = Analysis(
    ["main.py"],
    pathex=[str(ROOT / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)
a.datas = _drop_duplicate_ffmpeg(a.datas)
a.binaries = _drop_duplicate_ffmpeg(a.binaries)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="markface",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,          # GUI app: no console window
    disable_windowed_traceback=False,
    icon=str(ROOT / "assets" / "markface.ico") if (ROOT / "assets" / "markface.ico").exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="markface",
)
