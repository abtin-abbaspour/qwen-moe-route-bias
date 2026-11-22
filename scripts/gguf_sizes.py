"""Sum a GGUF's tensor bytes by role: routed experts, MTP (nextn) block, everything else."""
import sys, re, collections
sys.path.insert(0, sys.argv[2] if len(sys.argv) > 2 else "third_party/llama.cpp/gguf-py")
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
kv = {f.name: f for f in r.fields.values()}
def val(name):
    f = kv.get(name)
    return None if f is None else f.contents()
arch = val("general.architecture")
n_layer = val(f"{arch}.block_count")
n_nextn = val(f"{arch}.nextn_predict_layers") or 0
print("arch", arch, "block_count", n_layer, "nextn_layers", n_nextn,
      "experts", val(f"{arch}.expert_count"), "used", val(f"{arch}.expert_used_count"),
      "gating", val(f"{arch}.expert_gating_func"))
sizes = collections.Counter(); types = collections.defaultdict(collections.Counter)
first_mtp = (n_layer - n_nextn) if n_layer else 10**9
for t in r.tensors:
    m = re.match(r"blk\.(\d+)\.", t.name)
    layer = int(m.group(1)) if m else -1
    if layer >= first_mtp: role = "mtp_block"
    elif re.search(r"ffn_(up|down|gate|gate_up)_exps", t.name): role = "routed_experts"
    else: role = "other"
    sizes[role] += int(t.n_bytes); types[role][t.tensor_type.name] += int(t.n_bytes)
for k, v in sizes.items():
    print(f"{k:15s} {v/2**30:7.2f} GiB  {v/1e9:7.2f} GB  " + ", ".join(f"{n}:{b/2**30:.2f}" for n, b in types[k].most_common()))
