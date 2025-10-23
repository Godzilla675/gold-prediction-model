# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Gold Price Prediction Model GUI application.

This spec file includes special handling for XGBoost native libraries.
XGBoost requires its native library files (libxgboost.so on Linux, xgboost.dll on Windows)
to be explicitly included in the PyInstaller bundle, otherwise the built executable
will fail with "XGBoostLibraryNotFound" error at runtime.
"""
import os
import sys
from pathlib import Path

block_cipher = None

# Collect XGBoost library files
# Fix for XGBoost runtime error: "Cannot find XGBoost Library in the candidate path"
xgboost_libs = []
try:
    import xgboost
    xgb_path = Path(xgboost.__file__).parent
    lib_path = xgb_path / 'lib'
    
    if lib_path.exists():
        # Collect all library files from xgboost/lib directory
        # Expected files: libxgboost.so (Linux), xgboost.dll/libxgboost.dll (Windows), libxgboost.dylib (macOS)
        # These will be placed in xgboost/lib/ in the bundle
        for lib_file in lib_path.glob('*'):
            if lib_file.is_file():
                # Add to binaries with destination in xgboost/lib
                xgboost_libs.append((str(lib_file), 'xgboost/lib'))
                print(f"Including XGBoost library: {lib_file.name}")
except ImportError:
    print("Warning: XGBoost not found during spec file execution")

# Analysis - collect all necessary files and dependencies
a = Analysis(
    ['gui_app.py'],
    pathex=[],
    binaries=xgboost_libs,
    datas=[
        ('config.py', '.'),
        ('predict.py', '.'),
        ('data_fetcher.py', '.'),
        ('feature_engineering.py', '.'),
        ('models.py', '.'),
    ],
    hiddenimports=[
        'tkinter',
        'pandas',
        'numpy',
        'scikit-learn',
        'sklearn',
        'sklearn.preprocessing',
        'joblib',
        'tensorflow',
        'xgboost',
        'yfinance',
        'ta',
        'matplotlib',
        'seaborn',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='GoldPredictionModel',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # No console window for GUI app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # You can add an icon file here if available
)
