"""Smoke the full catalog: build every operator at its DEFAULT_SHAPE and run it
against its own torch golden on a real 910C device.

    python tests/test_all.py --device 0 [--op rmsnorm_forward]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pypto_ops  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--device", type=int, default=0)
    ap.add_argument("--op", type=str, default="")
    args = ap.parse_args()

    from golden import run

    names = [args.op] if args.op else pypto_ops.available_ops()
    n_pass = 0
    for name in names:
        mod = pypto_ops.get(name)
        fn, specs, golden_fn, rtol, atol = mod.build(mod.DEFAULT_SHAPE)
        result = run(fn=fn, specs=specs, golden_fn=golden_fn,
                     config=dict(backend_type="a2a3", device_id=args.device),
                     rtol=rtol, atol=atol)
        n_pass += bool(result.passed)
        print(f"{name:24s} {'PASS' if result.passed else 'FAIL: ' + str(result.error)}")
    print(f"\n{n_pass}/{len(names)} operators passed")
    sys.exit(0 if n_pass == len(names) else 1)


if __name__ == "__main__":
    main()
