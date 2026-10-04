"""Where does one training update go? Times a full update (lanes.chunk forward over `chunk` steps, backward,
optimizer) at several batch sizes, splits one update into phases, and prints the profiler's top ops.
    python scripts/moba/profile_update.py --data data/moba/full/dense --games 10-19 --init runs/token_wm/init --batches 128,384,768
"""
import argparse, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np, torch
import mobawm.wm.train as T
import mobawm.wm.data as D
from mobawm.wm.model import TokenWM

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/moba/full/dense"); ap.add_argument("--games", required=True)
ap.add_argument("--init", default="runs/token_wm/init"); ap.add_argument("--batches", default="128,384")
ap.add_argument("--chunk", type=int, default=16); ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--compile", action="store_true", help="torch.compile the model step")
a = ap.parse_args()
dev = torch.device("cuda"); T.BF16 = True
torch.backends.cuda.matmul.allow_tf32 = True
files = sorted((ROOT / a.data).glob("games.*.mbs"))
store = D.load_store(files, T.parse_games(a.games), dev)
sd = torch.load(ROOT / a.init / "stats.pt", map_location="cpu", weights_only=False)
stats = D.Stats(**sd["stats"]); cfg = sd["config"]
model = TokenWM(**cfg).to(dev); model.set_change_prior(stats.change_rate)
model.load_state_dict(torch.load(ROOT / a.init / "model.pt", map_location=dev), strict=False)
if a.compile:
    model.step = torch.compile(model.step, dynamic=False)
fz = D.Featurizer(dev); kind = torch.from_numpy(D.KIND).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
sync = torch.cuda.synchronize

def update(lanes):
    opt.zero_grad()
    loss, parts = lanes.chunk(model, fz, stats, kind, a.chunk)
    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return loss

for B in [int(x) for x in a.batches.split(",")]:
    gen = torch.Generator(device=dev); gen.manual_seed(0)
    try:
        lanes = T.Lanes(store, B, cfg.get("d", 128), gen, 4096, 6.0, 3.0, 64.0, 0.02)
        update(lanes); sync()                                           # warm-up (and compile)
        t0 = time.time()
        for _ in range(a.reps): update(lanes)
        sync(); dt = (time.time() - t0) / a.reps
        print(f"batch {B:5d}: {dt:6.2f} s/update | {B / dt:7.1f} lanes/s | peak GPU {torch.cuda.max_memory_allocated() / 1e9:5.1f} GB", flush=True)
    except torch.OutOfMemoryError:
        print(f"batch {B:5d}: out of GPU memory", flush=True); torch.cuda.empty_cache()
        break

# phase split for one update at the first batch size
B = int(a.batches.split(",")[0]); gen = torch.Generator(device=dev); gen.manual_seed(0)
lanes = T.Lanes(store, B, cfg.get("d", 128), gen, 4096, 6.0, 3.0, 64.0, 0.02); update(lanes); sync()
tm = {}
def wrap(name, fn):
    def w(*x, **k):
        sync(); t = time.time(); r = fn(*x, **k); sync(); tm[name] = tm.get(name, 0) + time.time() - t; return r
    return w
fz_orig, step_orig, loss_orig, ap_orig = D.Featurizer.__call__, model.step, T.step_loss, T.apply_prediction
D.Featurizer.__call__ = wrap("featurize", fz_orig); model.step = wrap("model.step", step_orig)
T.step_loss = wrap("step_loss", loss_orig); T.apply_prediction = wrap("apply_prediction", ap_orig)
sync(); t0 = time.time()
opt.zero_grad(); loss, _ = lanes.chunk(model, fz, stats, kind, a.chunk); sync(); t1 = time.time()
loss.backward(); sync(); t2 = time.time(); opt.step(); sync(); t3 = time.time()
tot = t3 - t0
print(f"\nphases of one update at batch {B} (forward over {a.chunk} steps {t1 - t0:.2f}s, backward {t2 - t1:.2f}s, optimizer {t3 - t2:.3f}s):")
for k, v in sorted(tm.items(), key=lambda kv: -kv[1]):
    print(f"   {k:18s} {v:6.2f}s  ({100 * v / tot:4.1f}%)")
print(f"   {'lanes bookkeeping':18s} {t1 - t0 - sum(tm.values()):6.2f}s")
D.Featurizer.__call__, model.step, T.step_loss, T.apply_prediction = fz_orig, step_orig, loss_orig, ap_orig
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    update(lanes); sync()
ev = prof.key_averages()
n_k = sum(e.count for e in ev if e.device_type == torch.autograd.DeviceType.CUDA)
cuda_t = sum(e.self_device_time_total for e in ev) / 1e6
print(f"\nprofiler: {n_k} GPU kernel launches per update, GPU busy {cuda_t:.2f}s of the update")
print(ev.table(sort_by="self_cpu_time_total", row_limit=12, max_name_column_width=40))
print(ev.table(sort_by="self_cuda_time_total", row_limit=15, max_name_column_width=60))
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True) as prof2:
    update(lanes); sync()
print(prof2.key_averages(group_by_stack_n=6).table(sort_by="self_cuda_time_total", row_limit=6, max_name_column_width=60))
