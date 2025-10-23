# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# Analysis - collect all necessary files and dependencies
a = Analysis(
    ['gui_app.py'],
    pathex=[],
    binaries=[],
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
