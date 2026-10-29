#!/usr/bin/env python3
"""Test tools/asm_audit.py on a real object file.

Assembles a tiny object whose hot-named function spills through vmovdqu32,
vmovdqu64 and movq stack stores. Three lines pin the spill rules: a folded-ALU
stack read, a GPR stack store and a register-register vmovdqa must not count.
Then runs the real CLI on it and checks the reported spills.

Run: python3 tools/test_asm_audit.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ASM = """\
        .text
        .globl  XoshiroSIMD_spill_test
XoshiroSIMD_spill_test:
        vmovdqu32 %ymm0, -32(%rbp)
        vmovdqu64 %ymm1, -64(%rbp)
        movq     %xmm2, -72(%rbp)
        mov      %eax, -8(%rbp)
        vmovdqa  %ymm0, %ymm1
        vpxor    -80(%rbp), %ymm3, %ymm4
        ret
"""


def find_toolchain() -> str | None:
    """Find an llvm-mc that has a matching llvm-objdump on PATH."""
    suffixes = [f"-{v}" for v in ("23", "24", "22", "21", "20", "19", "18")] + [""]
    for suf in suffixes:
        mc, od = f"llvm-mc{suf}", f"llvm-objdump{suf}"
        if shutil.which(mc) and shutil.which(od):
            return mc
    return None


def main() -> int:
    mc = find_toolchain()
    if mc is None:
        print("error: no llvm-mc/llvm-objdump pair on PATH", file=sys.stderr)
        return 1
    # asm_audit maps a clang-* compiler name to its llvm-objdump sibling.
    compiler = "clang" + mc.removeprefix("llvm-mc")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "t.s").write_text(ASM)
        obj = tmp / "t.o"
        subprocess.run(
            [mc, "--filetype=obj", str(obj.with_suffix(".s")), "-o", str(obj)],
            check=True,
        )
        report = tmp / "report.json"
        proc = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("asm_audit.py")),
                str(obj),
                "--compiler",
                compiler,
                "--json",
                str(report),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            print(
                f"error: asm_audit exited {proc.returncode}: {proc.stderr}",
                file=sys.stderr,
            )
            return 1
        rows = json.loads(report.read_text())

    assert len(rows) == 1, rows
    row = rows[0]
    assert row["function"] == "XoshiroSIMD_spill_test", row
    # vmovdqu32 + vmovdqu64 + movq stack stores. The mov GPR store, the
    # register-register vmovdqa and the folded vpxor read must not count.
    assert row["spills"] == 3, row
    assert row["width"] == "ymm", row
    assert not row["devectorised"], row
    print(f"ok: {row['function']} spills={row['spills']} width={row['width']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
