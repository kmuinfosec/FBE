import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import ints


def encode(Fn, cfg):
    sec = cfg["cnn"]
    ks, start, end, bins = sec.getint("token_limit"), sec.getint("log_start"), sec.getint("log_end"), sec.getint("log_bins")
    a, d = np.abs(Fn), np.sign(Fn)
    big = start + np.minimum(bins - 1, (np.log(np.maximum(a, start) / start) / np.log(end / start) * bins).astype(np.int64))
    b = np.where(a < start, a, big).astype(np.int64)
    tok = (d + 1).astype(np.int64) * ks + b
    cont = np.stack([d, np.log1p(a) / 10, (Fn != 0)], 1).astype(np.float32)
    return torch.from_numpy(tok), torch.from_numpy(cont)


class Net(nn.Module):
    def __init__(self, nc, cfg):
        super().__init__()
        sec = cfg["cnn"]
        self.dil = ints(cfg, "cnn", "dilations")
        ch, emb = sec.getint("channels"), sec.getint("embedding_dim")
        self.emb = nn.Embedding(3 * sec.getint("token_limit"), emb)
        cin = emb + 3
        self.convs, self.bns = nn.ModuleList(), nn.ModuleList()
        for d in self.dil:
            self.convs.append(nn.Conv1d(cin, ch, 3, dilation=d, padding=d))
            self.bns.append(nn.BatchNorm1d(ch))
            cin = ch
        self.drop = nn.Dropout(sec.getfloat("dropout"))
        self.head = nn.Linear(2 * ch, nc)

    def forward(self, tok, cont):
        h = torch.cat([self.emb(tok).transpose(1, 2), cont], 1)
        m = cont[:, 2:3]
        for conv, bn, d in zip(self.convs, self.bns, self.dil):
            h = F.relu(bn(conv(h)))
            m = F.max_pool1d(F.pad(m, (d, d)), 3, 1, dilation=d)
        m = m > 0
        hmax = h.masked_fill(~m, -1e4).amax(2)
        hmean = (h * m).sum(2) / m.sum(2).clamp(min=1)
        hmax = hmax.masked_fill(~m.any(2), 0)
        return self.head(self.drop(torch.cat([hmax, hmean], 1)))


def device(cfg):
    name = cfg.get("cnn", "device", fallback="auto")
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def fit(cfg, dataset, n, seed, Fn, y, nc):
    sec = cfg["cnn"]
    dev = device(cfg)
    torch.set_num_threads(cfg.getint("experiment", "threads"))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    small = len(Fn) < sec.getint("small_threshold")
    batch = sec.getint("small_batch") if small else sec.getint("large_batch")
    epochs = sec.getint("small_epochs") if small else sec.getint("large_epochs")
    tok, cont = encode(Fn, cfg)
    yt = torch.from_numpy(y.astype(np.int64))
    net = Net(nc, cfg).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=sec.getfloat("lr"), weight_decay=sec.getfloat("weight_decay"))
    steps = epochs * ((len(Fn) + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=sec.getfloat("lr"), total_steps=steps, pct_start=0.1)
    for _ in range(epochs):
        net.train()
        perm = torch.from_numpy(rng.permutation(len(Fn)))
        for s in range(0, len(Fn), batch):
            b = perm[s:s + batch]
            loss = F.cross_entropy(net(tok[b].to(dev), cont[b].to(dev)), yt[b].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    net.eval()
    net.cfg, net.dev = cfg, dev
    return net


def predict(net, Fn):
    eb = net.cfg.getint("cnn", "eval_batch")
    with torch.no_grad():
        out = []
        for s in range(0, len(Fn), eb):
            tok, cont = encode(Fn[s:s + eb], net.cfg)
            out.append(net(tok.to(net.dev), cont.to(net.dev)).argmax(1).cpu().numpy())
        return np.concatenate(out)
