import argparse
import sys
import time

import torch

from exact_decimal_adder_transformer import ExactDecimalAdderTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Randomized verification for ExactDecimalAdderTransformer.")
    parser.add_argument("--total", type=int, default=100_000_000, help="Total number of test cases.")
    parser.add_argument("--batch", type=int, default=1_000_000, help="Batch size per iteration.")
    parser.add_argument("--digits", type=int, default=10, help="Number of digits per addend.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.total <= 0:
        print("ERROR: --total must be positive.")
        return 1
    if args.batch <= 0:
        print("ERROR: --batch must be positive.")
        return 1
    if args.digits <= 0:
        print("ERROR: --digits must be positive.")
        return 1

    model = ExactDecimalAdderTransformer(n_digits=args.digits)
    if model.count_trainable_parameters() != 0:
        print("ERROR: model has trainable parameters.")
        return 1

    rng = torch.Generator(device="cpu")
    rng.manual_seed(args.seed)

    total = args.total
    batch = args.batch
    checked = 0
    start = time.time()

    num_batches = (total + batch - 1) // batch
    for i in range(num_batches):
        bs = min(batch, total - checked)

        a = torch.randint(0, 10 ** args.digits, (bs,), dtype=torch.int64, generator=rng)
        b = torch.randint(0, 10 ** args.digits, (bs,), dtype=torch.int64, generator=rng)

        a_digits = model.int_to_lsb_digits(a, n_digits=args.digits)
        b_digits = model.int_to_lsb_digits(b, n_digits=args.digits)

        out_digits = model(a_digits, b_digits)
        out_int = model.lsb_digits_to_int(out_digits)

        expected = a + b

        if checked == 0 and bs > 0:
            print(f"First problem: {a[0].item()} + {b[0].item()} = {out_int[0].item()}")
        if checked + bs == total and bs > 0:
            print(f"Last problem: {a[-1].item()} + {b[-1].item()} = {out_int[-1].item()}")

        if not torch.equal(out_int, expected):
            mismatch_idx = torch.nonzero(out_int != expected, as_tuple=False)[0].item()
            print("FAIL")
            print("index", checked + mismatch_idx)
            print("a", a[mismatch_idx].item())
            print("b", b[mismatch_idx].item())
            print("got", out_int[mismatch_idx].item())
            print("expected", expected[mismatch_idx].item())
            return 1

        checked += bs
        if (i + 1) % 10 == 0 or checked == total:
            elapsed = time.time() - start
            rate = checked / max(elapsed, 1e-9)
            print(f"checked {checked}/{total} ({rate:,.0f} cases/s)")

    elapsed = time.time() - start
    print(f"PASS: {total} cases in {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

