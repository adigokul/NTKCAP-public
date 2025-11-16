#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
check_cygnus_kernel.py

Check / start / stop / watch CygnusKernel.exe on Windows.

Usage (cmd.exe):
    cd /d C:\Users\MyUser\Desktop\NTKCAP\cnimes_RT
    python check_cygnus_kernel.py status
    python check_cygnus_kernel.py start
    python check_cygnus_kernel.py stop
    python check_cygnus_kernel.py watch --interval 2

The script prefers `psutil` if installed; otherwise it falls back to
`tasklist`/`taskkill` system commands.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Union

# Default executable path (absolute). Adjust if your installation differs.
DEFAULT_EXE = Path(r"C:\Users\MyUser\Desktop\NTKCAP\cnimes_RT\Cygnus_Kernel_0.13.0.2\Cygnus_Kernel_0.13.0.2\core\CygnusKernel.exe")
DEFAULT_NAME = "CygnusKernel.exe"

try:
    import psutil  # optional, provides richer process control
    HAS_PSUTIL = True
except Exception:
    psutil = None
    HAS_PSUTIL = False


def is_running_by_name(exe_name: str = DEFAULT_NAME) -> List:
    """Return list of psutil.Process objects (if psutil) or list of PIDs (ints/strings).

    If psutil is not available, returns a list of PID strings found by `tasklist`.
    """
    if HAS_PSUTIL:
        matches = []
        for p in psutil.process_iter(['pid', 'name', 'exe']):
            try:
                name = (p.info.get('name') or '').lower()
                exe = (p.info.get('exe') or '')
                if name == exe_name.lower():
                    matches.append(p)
                elif exe and Path(exe).name.lower() == exe_name.lower():
                    matches.append(p)
            except Exception:
                continue
        return matches
    else:
        try:
            out = subprocess.check_output(['tasklist', '/FI', f'IMAGENAME eq {exe_name}'], stderr=subprocess.DEVNULL, text=True, encoding='utf-8')
            lines = [l for l in out.splitlines() if l.strip()]
            if any('no tasks are running' in l.lower() for l in lines):
                return []
            matches: List[str] = []
            for l in lines:
                if l.lower().startswith(exe_name.lower()):
                    parts = l.split()
                    if len(parts) >= 2:
                        matches.append(parts[1])
            return matches
        except subprocess.CalledProcessError:
            return []


def is_running_by_path(exe_path: Path) -> List:
    if not exe_path:
        return []
    if HAS_PSUTIL:
        matches = []
        try:
            target = exe_path.resolve()
        except Exception:
            target = exe_path
        for proc in psutil.process_iter(['pid', 'exe', 'name']):
            try:
                exe = proc.info.get('exe')
                if not exe:
                    continue
                try:
                    if Path(exe).resolve() == target:
                        matches.append(proc)
                except Exception:
                    if Path(exe) == exe_path:
                        matches.append(proc)
            except Exception:
                continue
        return matches
    else:
        return is_running_by_name(exe_path.name)


def start_exe(exe_path: Path) -> int:
    if not exe_path.exists():
        raise FileNotFoundError(f"Executable not found: {exe_path}")
    # Start the exe without capturing output and without creating an extra shell
    p = subprocess.Popen([str(exe_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=False)
    return p.pid


def stop_by_name(exe_name: str = DEFAULT_NAME) -> List[int]:
    if HAS_PSUTIL:
        stopped: List[int] = []
        for p in is_running_by_name(exe_name):
            try:
                p.terminate()
                p.wait(5)
                stopped.append(p.pid)
            except Exception:
                try:
                    p.kill()
                    stopped.append(p.pid)
                except Exception:
                    pass
        return stopped
    else:
        try:
            subprocess.check_call(['taskkill', '/IM', exe_name, '/F'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return [0]
        except subprocess.CalledProcessError:
            return []


def stop_by_path(exe_path: Path) -> List[int]:
    if HAS_PSUTIL:
        stopped: List[int] = []
        for p in is_running_by_path(exe_path):
            try:
                p.terminate()
                p.wait(5)
                stopped.append(p.pid)
            except Exception:
                try:
                    p.kill()
                    stopped.append(p.pid)
                except Exception:
                    pass
        return stopped
    else:
        return stop_by_name(exe_path.name)


def pretty_report(matches: List[Union['psutil.Process', int]]) -> str:
    if not matches:
        return "Not running"
    if HAS_PSUTIL:
        s = []
        for p in matches:
            try:
                cmd = ' '.join(p.cmdline() or [])
                s.append(f"pid={p.pid}, name={p.name()}, cmdline={cmd}")
            except Exception:
                s.append(f"pid={getattr(p, 'pid', '?')}")
        return "Running: " + "; ".join(s)
    else:
        return "Running (PIDs): " + ",".join(str(x) for x in matches)


def main() -> None:
    parser = argparse.ArgumentParser(description='Check/start/stop CygnusKernel.exe')
    parser.add_argument('action', choices=['status', 'start', 'stop', 'watch'], help='Action to perform')
    parser.add_argument('--exe', default=str(DEFAULT_EXE), help='Path to CygnusKernel.exe')
    parser.add_argument('--name', default=DEFAULT_NAME, help='Process image name')
    parser.add_argument('--interval', type=float, default=1.0, help='Watch interval in seconds')
    args = parser.parse_args()

    exe_path = Path(args.exe)

    if args.action == 'status':
        matches_path = is_running_by_path(exe_path)
        if matches_path:
            print(pretty_report(matches_path))
            sys.exit(0)
        matches_name = is_running_by_name(args.name)
        print(pretty_report(matches_name))
        sys.exit(0 if matches_name else 2)

    if args.action == 'start':
        if not exe_path.exists():
            print(f"ERROR: exe not found: {exe_path}")
            sys.exit(3)
        if is_running_by_path(exe_path) or is_running_by_name(args.name):
            print("Already running")
            sys.exit(0)
        try:
            pid = start_exe(exe_path)
            print(f"Started {exe_path} (pid {pid})")
            sys.exit(0)
        except Exception as e:
            print(f"Failed to start: {e}")
            sys.exit(4)

    if args.action == 'stop':
        stopped = stop_by_path(exe_path)
        if stopped:
            print(f"Stopped: {stopped}")
            sys.exit(0)
        else:
            print("No matching process found to stop")
            sys.exit(5)

    if args.action == 'watch':
        try:
            print(f"Watching {args.name} (press Ctrl-C to quit) ...")
            while True:
                matches_path = is_running_by_path(exe_path)
                matches_name = is_running_by_name(args.name)
                running = bool(matches_path or matches_name)
                ts = time.strftime('%Y-%m-%d %H:%M:%S')
                print(f"[{ts}] {'RUNNING' if running else 'NOT RUNNING'}")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print('\nWatch stopped by user')
            sys.exit(0)


if __name__ == '__main__':
    main()
