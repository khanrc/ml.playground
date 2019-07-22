""" Pytorch GPU memory checklist:
1. Do not use accumulate model output itself
    - use `item()`
2. Remove model output variable for free forward features
    - you can use `del` (= re-alloc var) or `detach()` ...

Ref: https://pytorch.org/docs/stable/notes/faq.html
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def check_mem(tag):
    print("{:15s}: {:.0f}k / {:.0f}k".format(
        tag, torch.cuda.memory_allocated() / 1024, torch.cuda.memory_cached() / 1024))

docs = """Now we will see pytorch memory usage using
torch.cuda.memory_allocated() and torch.cuda.memory_cached().
"""
print(docs)

check_mem("start")

net = nn.Sequential(
    nn.Linear(1024, 1024),
    nn.Linear(1024, 1024),
    nn.Linear(1024, 1024),
    nn.Linear(1024, 10)
).cuda()

check_mem("net")
B = 128

print("### Task 1 ###")
accum = 0.
for i in range(3):
    x = torch.rand(B, 1024).cuda()

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

# XXX: the memory allocated by backward() cannot cleanup.
# 처음 backward 를 하는 시점에서 backward memory 가 확보되며,
# 그 이후로 지울 수 없다.
print("\n### Task 2. backward() ###")
for i in range(3):
    x = torch.rand(B, 1024).cuda()
    #y = torch.randint(0, 10, [128]).cuda()
    y = torch.rand(B, 10).cuda()

    check_mem(f"{i}: make x, y")
    r = net(x)
    check_mem(f"{i}: forward")

    loss = F.l1_loss(r, y)
    check_mem(f"{i}: calc loss")

    loss.backward()
    check_mem(f"{i}: backward")

    # deletion
    del loss
    check_mem(f"{i}: del loss")
    del r
    check_mem(f"{i}: del r")
    net.zero_grad()
    check_mem(f"{i}: zero_grad")

    del x, y
    check_mem(f"{i}: del x, y")
    torch.cuda.empty_cache()
    check_mem(f"{i}: empty cache")
    print("")


def forward_backward(net):
    x = torch.rand(B, 1024).cuda()
    y = torch.rand(B, 10).cuda()
    r = net(x)
    loss = F.l1_loss(r, y)
    loss.backward()


print("\n### Task 3. func-backward() ###")
for i in range(3):
    check_mem(f"{i}: before fb")
    forward_backward(net)
    check_mem(f"{i}: after fb")

    net.zero_grad()
    torch.cuda.empty_cache()
    check_mem(f"{i}: please")
    print("")
