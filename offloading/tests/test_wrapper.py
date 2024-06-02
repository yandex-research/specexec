import torch
import transformers

from offloading.storage_wrapper import ModuleWithStorage


def test_wrapper():
    model_name = "meta-llama/Llama-2-7b-hf"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype='auto', device_map='cuda:0')

    la = ModuleWithStorage(model.model.layers[22])
    lb = ModuleWithStorage(model.model.layers[23])
    x = torch.randn(1, 1, 4096).cuda().half()

    ya, = la(x)
    yb, = lb(x)
    assert not torch.all(ya == yb)

    la.storage.copy_(lb.storage);

    ya_new, = la(x)
    yb_new, = lb(x)
    assert not torch.all(ya_new == ya)
    assert torch.all(ya_new == yb_new)
    assert torch.all(yb_new == yb)



