#!/usr/bin/env python3
"""
mycelium.py — the orchestrator. the connective tissue between organisms.

molequla has four elements:
    molequla.go   — Go
    molequla.c    — C
    molequla.js   — JavaScript
    molequla.rs   — Rust (the mouth)

mycelium connects them. it reads the field (mesh.db), computes
system-level awareness via METHOD (C-native, BLAS-accelerated),
and writes steering deltas for the mouth to consume.

usage:
    python3 mycelium.py                    # default mesh.db in current dir
    python3 mycelium.py --mesh ./mesh.db   # explicit path
    python3 mycelium.py --interval 2.0     # step every 2 seconds
    python3 mycelium.py --once             # single step, print, exit

part of molequla. the method that connects organisms.
"""

import argparse
import asyncio
import json
import os
import signal
import sqlite3
import sys
import time
from pathlib import Path

# Add parent dir so ariannamethod is importable
sys.path.insert(0, str(Path(__file__).parent))

from ariannamethod import Method


# ═══════════════════════════════════════════════════════════════════════════════
# field monitor — watches organism health
# ═══════════════════════════════════════════════════════════════════════════════

class FieldMonitor:
    """tracks field health over time, detects anomalies."""

    def __init__(self):
        self.history = []       # list of steering dicts
        self.max_history = 64
        self.alerts = []

    def record(self, steering):
        self.history.append(steering)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        self._check_alerts(steering)

    def _check_alerts(self, s):
        self.alerts.clear()

        if s.get("n_organisms", 0) == 0:
            self.alerts.append("no organisms alive")
            return

        if s.get("coherence", 1.0) < 0.2:
            self.alerts.append(f"coherence critical: {s['coherence']:.3f}")

        if s.get("entropy", 0) > 3.0:
            self.alerts.append(f"entropy explosion: {s['entropy']:.3f}")

        action = s.get("action", "")
        if len(self.history) >= 8:
            recent_actions = [h.get("action") for h in self.history[-8:]]
            if all(a == "dampen" for a in recent_actions):
                self.alerts.append("stuck in dampen loop")
            if all(a == "realign" for a in recent_actions):
                self.alerts.append("persistent incoherence")

    def status_line(self, steering):
        n = steering.get("n_organisms", 0)
        action = steering.get("action", "?")
        strength = steering.get("strength", 0)
        ent = steering.get("entropy", 0)
        syn = steering.get("syntropy", 0)
        coh = steering.get("coherence", 0)
        step = steering.get("step", 0)

        line = (f"[mycelium] step={step} organisms={n} "
                f"action={action}({strength:.2f}) "
                f"H={ent:.3f} S={syn:.3f} C={coh:.3f}")

        if self.alerts:
            line += f"  !! {'; '.join(self.alerts)}"

        return line


# ═══════════════════════════════════════════════════════════════════════════════
# drift tracker — which organisms are diverging
# ═══════════════════════════════════════════════════════════════════════════════

class DriftTracker:
    """identifies organisms drifting from the field mean."""

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.drifters = {}

    def update(self, method):
        self.drifters = method.field_drift()
        return self.drifters

    def report(self):
        if not self.drifters:
            return None
        lines = [f"  organism {oid}: deviation={dev:.3f}"
                 for oid, dev in sorted(self.drifters.items(), key=lambda x: -x[1])]
        return "[mycelium] drifters detected:\n" + "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# mycelium loop
# ═══════════════════════════════════════════════════════════════════════════════

class Mycelium:
    """the orchestrator. connects organisms through METHOD."""

    def __init__(self, mesh_path="mesh.db", interval=1.0, verbose=True):
        self.method = Method(mesh_path)
        self.interval = interval
        self.verbose = verbose
        self.monitor = FieldMonitor()
        self.drift = DriftTracker()
        self.running = False
        self._step_count = 0

    def step(self):
        """one tick: read field → METHOD → monitor → report."""
        steering = self.method.step(dt=self.interval)
        self.monitor.record(steering)
        self._step_count += 1

        # Check for drifters every 4 steps
        drift_report = None
        if self._step_count % 4 == 0:
            self.drift.update(self.method)
            drift_report = self.drift.report()

        if self.verbose:
            print(self.monitor.status_line(steering))
            if drift_report:
                print(drift_report)

        return steering

    async def run(self):
        """async loop: step every interval seconds."""
        self.running = True
        lib_status = "C+BLAS" if self.method.lib else "Python fallback"
        print(f"[mycelium] started. METHOD engine: {lib_status}")
        print(f"[mycelium] mesh: {self.method.mesh_path}")
        print(f"[mycelium] interval: {self.interval}s")
        print()

        while self.running:
            try:
                self.step()
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"[mycelium] error: {e}")

            await asyncio.sleep(self.interval)

        print("\n[mycelium] stopped.")

    def stop(self):
        self.running = False


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="mycelium — distributed cognition orchestrator")
    parser.add_argument("--mesh", default="mesh.db",
                        help="path to mesh.db (default: ./mesh.db)")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="seconds between METHOD steps (default: 1.0)")
    parser.add_argument("--once", action="store_true",
                        help="single step, print JSON, exit")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress per-step output")
    args = parser.parse_args()

    myc = Mycelium(
        mesh_path=args.mesh,
        interval=args.interval,
        verbose=not args.quiet,
    )

    if args.once:
        steering = myc.step()
        print(json.dumps(steering, indent=2))
        return

    # Handle SIGINT/SIGTERM gracefully
    def handle_signal(sig, frame):
        myc.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    asyncio.run(myc.run())


if __name__ == "__main__":
    main()
