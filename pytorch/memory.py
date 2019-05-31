""" Pytorch GPU memory checklist:
1. Do not use accumulate model output itself
    - use `item()`
2. Remove model output variable for free forward features
    - you can use `del` (= re-alloc var) or `detach()` ...

Ref: https://pytorch.org/docs/stable/notes/faq.html
"""

import torch
import torch.nn as nn


def check_mem(tag):
    print("{:15s}: {:.0f}k / {:.0f}k".format(
        tag, torch.cuda.memory_allocated() / 1024, torch.cuda.memory_cached() / 1024))


check_mem("start")

net = nn.Sequential(
    nn.Linear(1024, 1024),
    nn.Linear(1024, 1024),
    nn.Linear(1024, 1024)
).cuda()

check_mem("net")


accum = 0.
for i in range(3):
    x = torch.rand(128, 1024).cuda()

    check_mem(f"{i}: x")

    r = net(x)

    check_mem(f"{i}: forward")

    accum += r.sum().item() # [!] must use `item()`
    #del r # [!] del tensor => free GPU memory

    # [!] detach tensor => free forward features (maybe)
    # detach vs del: detach free features only / del free features + results.
    r = r.sum().detach()
    # re-alloc variable like above is same as del

    check_mem(f"{i}: del result")
