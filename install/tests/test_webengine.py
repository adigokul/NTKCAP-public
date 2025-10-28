#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

try:
    from PyQt6.QtCore import PYQT_VERSION_STR
    print(f"PyQt6 version: {PYQT_VERSION_STR}")
except ImportError as e:
    print(f"PyQt6 import error: {e}")
    sys.exit(1)

try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    print("✓ WebEngine import successful")
except ImportError as e:
    print(f"✗ WebEngine import failed: {e}")
    
    # 檢查是否有 WebEngine 相關套件
    try:
        import PyQt6.QtWebEngineCore
        print("  - QtWebEngineCore available")
    except ImportError:
        print("  - QtWebEngineCore not available")
        
    try:
        import PyQt6.QtWebEngineWidgets
        print("  - QtWebEngineWidgets available")
    except ImportError:
        print("  - QtWebEngineWidgets not available")
        
except Exception as e:
    print(f"✗ Unexpected WebEngine error: {e}")
    import traceback
    traceback.print_exc()

# 檢查已安裝的 PyQt6 相關套件
try:
    import pkg_resources
    installed_packages = [pkg.project_name for pkg in pkg_resources.working_set]
    pyqt_packages = [pkg for pkg in installed_packages if 'pyqt' in pkg.lower()]
    print(f"\nInstalled PyQt packages: {pyqt_packages}")
except:
    pass