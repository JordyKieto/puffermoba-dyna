"""Checkpoint save / load / resume must actually restore training state.

Weight equality alone is not enough: a resume that drops optimizer moments or
scheduler position silently changes the training trajectory. These check the
full triple, plus the contract guard that stops a model being loaded against a
state layout it was not trained on.

    python3 tests/test_checkpoint.py
"""
import shutil
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from racewm import checkpoint as ckpt_mod  # noqa: E402
from racewm import contract  # noqa: E402

FAILURES = []


def check(cond, name):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        FAILURES.append(name)


def make():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 16), nn.SiLU(), nn.Linear(16, 4))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
    return model, opt, sch


def train_steps(model, opt, sch, n, seed):
    g = torch.Generator().manual_seed(seed)
    for _ in range(n):
        x = torch.randn(32, 8, generator=g)
        y = torch.randn(32, 4, generator=g)
        loss = nn.functional.mse_loss(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sch.step()
    return float(loss.detach())


def main():
    root = Path(tempfile.mkdtemp())
    try:
        print("== save / resume round trip ==")
        model, opt, sch = make()
        cp = ckpt_mod.Checkpointer(root, "run_a", config={"hidden": 16})
        train_steps(model, opt, sch, 5, seed=1)
        cp.save(step=5, model=model, optimizer=opt, scheduler=sch,
                metrics={"val_nmse": 0.5})

        # Reference: keep training the original.
        ref_loss = train_steps(model, opt, sch, 5, seed=2)

        # Resumed: fresh objects, restored, then the identical 5 steps.
        m2, o2, s2 = make()
        cp2 = ckpt_mod.Checkpointer(root, "run_a")
        step = cp2.resume(m2, o2, s2)
        check(step == 5, "resume returns the saved step")
        res_loss = train_steps(m2, o2, s2, 5, seed=2)

        check(abs(ref_loss - res_loss) < 1e-9,
              "resumed run reproduces the uninterrupted run exactly")
        same = all(torch.equal(a, b) for a, b in
                   zip(model.state_dict().values(), m2.state_dict().values()))
        check(same, "weights identical after resume + equal training")
        check(abs(o2.param_groups[0]["lr"] - opt.param_groups[0]["lr"]) < 1e-12,
              "scheduler position restored (lr matches)")

        print("== best/latest tracking ==")
        model, opt, sch = make()
        cp = ckpt_mod.Checkpointer(root, "run_b", metric_name="val_nmse", metric_mode="min")
        cp.save(step=1, model=model, optimizer=opt, metrics={"val_nmse": 0.9})
        cp.save(step=2, model=model, optimizer=opt, metrics={"val_nmse": 0.3})
        cp.save(step=3, model=model, optimizer=opt, metrics={"val_nmse": 0.7})
        man = ckpt_mod.read_manifest(root, "run_b")
        check(man.best_step == 2, "best.pt tracks the best metric, not the last")
        check((root / "run_b" / "latest.pt").exists(), "latest.pt written")
        check(len(man.history) == 3, "manifest records every save")

        print("== contract guard ==")
        real = contract.schema_hash
        m3, _, _ = make()
        try:
            contract.schema_hash = lambda: "deadbeefdeadbeef"
            try:
                ckpt_mod.load_into(root / "run_b" / "best.pt", m3)
                check(False, "loading under a changed contract is refused")
            except RuntimeError:
                check(True, "loading under a changed contract is refused")
            try:
                ckpt_mod.Checkpointer(root, "run_b")
                check(False, "reopening a run under a changed contract is refused")
            except RuntimeError:
                check(True, "reopening a run under a changed contract is refused")
        finally:
            contract.schema_hash = real

        print("== resume with no checkpoint ==")
        m4, o4, _ = make()
        cp4 = ckpt_mod.Checkpointer(root, "run_fresh")
        check(cp4.resume(m4, o4) == 0, "resume on an empty run returns step 0")
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print()
    if FAILURES:
        print(f"checkpoint: {len(FAILURES)} FAILURE(S): {FAILURES}")
        return 1
    print("checkpoint: ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
