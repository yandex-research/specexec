import torch
import transformers

import sys
from pathlib import Path

# Add the parent directory to sys.path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))


from storage_wrapper import ModuleWithStorage


model_name = "TheBloke/Llama-2-7B-chat-GPTQ"
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype='auto', device_map='cuda:0')

la = ModuleWithStorage(model.model.layers[22])
lb = ModuleWithStorage(model.model.layers[23])
x = torch.randn(1, 1, 4096).cuda().half()

la_0 = model.model.layers[22]
k0 = la_0.self_attn.k_proj
k1 = la.module.self_attn.k_proj

y0 = k0(x)
y1 = k1(x)
y0 = k0(x)
y1 = k1(x)

ya, = la(x)
yb, = lb(x)
assert not torch.all(ya == yb)

la.storage.copy_(lb.storage)

ya_new, = la(x)
yb_new, = lb(x)
assert not torch.all(ya_new == ya)
assert torch.all(ya_new == yb_new)
assert torch.all(yb_new == yb)
