# DazArgonMiner

NVIDIA CUDA miner for **Argonmine (ARG)**: Argon2id with t=1, m=8 MiB, p=1. It runs on **HiveOS** as a custom miner, and on **Ubuntu** and **WSL2**.

- **Fast kernel**: one warp per hash, with the previous block kept in registers. On an RTX 5080 it is up to 30 % faster than the original kernel at the same batch size, and 7 % faster than the original kernel at its own best batch (see [Performance](#performance)).
- **fast2 kernel** (new in 0.6.0): the permutation runs in registers with warp shuffles, and in the first half of each hash the next reference block is prefetched while the current one is computed. Each GPU measures fast2 against fast and keeps the faster one.
- **Self-tested hashes**: every GPU checks its kernel against a known-answer Argon2id tag at start-up and falls back fast2 → fast → original if the result is wrong.
- **Kernel and batch tuned per GPU**: on first start each card tests the kernels, then 3–4 batch sizes, 45 s each while mining, keeps the fastest pair and saves it.
- **Multi-GPU rigs**: one process per GPU, one shared work fetcher, a separate nonce range per card.
- **Dashboard**: a fixed screen refreshed every 2 s with every GPU, the **estimated time to find a block**, the **24 h chance**, the estimated network hashrate, the current round, your blocks and recent events.
- **HiveOS integration**: hashrate, temperatures, fans and accepted/rejected blocks show up on the dashboard.
- **Low CPU use**: the CPU sleeps while the GPU works, which matters on rigs with small CPUs.

> **Solo mining.** Argonmine has no stratum pool for native miners. Each block you find pays you its whole reward, and between blocks you earn nothing. Rewards are spendable after 100 confirmations.

---

## Contents

- [Requirements](#requirements)
- [HiveOS](#hiveos)
- [Ubuntu / WSL2](#ubuntu--wsl2)
- [Console](#console)
- [Options](#options)
- [Performance](#performance)
- [Overclocking](#overclocking)
- [Dev fee](#dev-fee)
- [How it works](#how-it-works)
- [Troubleshooting](#troubleshooting)
- [Credits and license](#credits-and-license)

---

## Requirements

| | |
|---|---|
| GPU | NVIDIA RTX 30 (sm_86), RTX 40 (sm_89), RTX 50 (sm_120) |
| VRAM | 8 MiB per hash in flight. The batch is picked to leave about 1.5 GiB free; 8 GB cards run fine |
| Driver | A recent driver. 570 or newer is recommended, and RTX 50 cards require it |
| System | HiveOS, or Ubuntu 22.04/24.04 (native or WSL2), x86_64 |
| Tools | `python3`. `g++` only if the miner has to compile itself (see below) |

**Prebuilt binary**: both release archives ship a ready-to-run `bin/argon_gpu`, built on HiveOS with CUDA 12.9 for sm_86, sm_89 and sm_120 (+ PTX). It needs **glibc 2.34 or newer**: HiveOS, Ubuntu 22.04, Ubuntu 24.04 and WSL2 all qualify. `BUILD_INFO.txt` in the archive gives the details.

**Automatic fallback**: if the bundled binary cannot run on a system (for example an older glibc), the miner compiles itself at start-up, which takes 1–3 minutes. If `nvcc` 12.8 or newer is installed, it is used. Otherwise the official NVIDIA CUDA redistributables (`cuda_nvcc`, `cuda_cudart`, `cuda_cccl`, about 100 MB) are downloaded from `developer.download.nvidia.com` into the miner's own folder, and their SHA-256 is checked against NVIDIA's manifest. No `pip` or `apt` package is installed.

---

## HiveOS

Create a flight sheet with the **Custom** miner:

| Field | Value |
|---|---|
| Coin | `ARG` (create it if needed, the name is free) |
| Wallet | your `arg1...` address |
| Pool | Configure in miner |
| Miner name | `argonmine-cuda` |
| Installation URL | `https://github.com/djdaz31/DazArgonMiner/releases/download/0.6.0/argonmine-cuda-0.6.0.tar.gz` |
| Hash algorithm | `argon2id` |
| Wallet and worker template | `%WAL%` (address only, no `.%WORKER_NAME%`) |
| Pool URL | `https://explorer.argonmine.com` |
| Pass | *(empty)* |
| Extra config arguments | *(empty)*, or for example `devices=0,1,2 retune=1` |

Nothing is compiled: the first start shows `[selftest] fast kernel: OK`, then kernel and batch tuning (3–5 min), then the dashboard. Follow it with `miner log`.

**Extra config options**

| Option | Effect |
|---|---|
| `devices=0,2,3` | mine only on these GPU indices (default: all) |
| `batch=768` | force one batch size on every GPU and skip tuning |
| `retune=1` | test kernels and batch sizes again at the next start (after an OC change). Remove it afterwards |
| `kernel=fast` | force one kernel on every GPU: `fast2`, `fast` or `orig` |
| `tune=60` | seconds per batch size while tuning (default 45) |

### Example flight sheet

> ⚠️ This example mines to the **author's** address. Put **your own** `arg1...` address in the wallet, or the blocks you find go to the author.

```
Flight sheet name : ARG argonmine-cuda
Coin              : ARG
Wallet            : arg1q56234qn7f2mym8sjj7m4uww2aghgh8pah4q6gnnpusrutfqkn45shvc7tz   ← replace
Pool              : Configure in miner
Miner             : Custom
  Miner name                 : argonmine-cuda
  Installation URL           : https://github.com/djdaz31/DazArgonMiner/releases/download/0.6.0/argonmine-cuda-0.6.0.tar.gz
  Hash algorithm             : argon2id
  Wallet and worker template : %WAL%
  Pool URL                   : https://explorer.argonmine.com
  Pass                       : (empty)
  Extra config arguments     : (empty)
```

<!-- screenshot of the real flight sheet: put it in docs/flightsheet.png and uncomment
![HiveOS flight sheet](docs/flightsheet.png)
-->

**Matching OC profile** (algorithm `argon2id`) measured on a rig of 4× RTX 3070 + 1× RTX 4070 (GPU order: 3070, 4070, 3070, 3070, 3070). Total **122.6 kH/s**:

| Field | Value |
|---|---|
| Core Clock Offset | `200 100 0 200 200` |
| Lock Core Clock | `0 2700 0 0 0` |
| Memory Clock Offset | `1000 800 1000 1000 1000` |
| Lock Memory Clock | *(empty, important)* |
| Power Limit | `150 180 150 150 150` |

These are starting values, not limits: the 3070s were still power-limited at 150 W. See [Overclocking](#overclocking).

**Reinstalling**: `blocks.json` (your block history) and `tune.json` (saved batches) live in `/hive/miners/custom/argonmine-cuda/`. Copy them somewhere safe before deleting that folder. If the miner ever had to compile itself, keep the `cuda/` sub-folder too, so the compiler is not downloaded again.

---

## Ubuntu / WSL2

Use the **linux** archive, which is prebuilt, so nothing is compiled:

```bash
sudo apt install -y python3 wget tmux
mkdir -p ~/argonmine && cd ~/argonmine
wget https://github.com/djdaz31/DazArgonMiner/releases/download/0.6.0/argonmine-cuda-linux-0.6.0.tar.gz
tar xzf argonmine-cuda-linux-0.6.0.tar.gz && cd argonmine-cuda
./bin/argon_gpu --selftest          # must print "fast kernel: OK" and "original kernel: OK"
```

Run it inside `tmux`, so it keeps going when the terminal closes:

```bash
tmux new -s arg
./run.sh arg1YOURADDRESS
```

To detach press `Ctrl+B` then `D`, and to come back run `tmux attach -t arg`.

- `run.sh` passes any extra option to the miner, for example `./run.sh arg1... --devices 0 --retune`. See [Options](#options).
- If the prebuilt binary cannot run on your system, `run.sh` compiles it once (this needs `g++`: `sudo apt install -y g++`).
- If several machines share one public IP, add `--interval 4` on the extra ones to space out requests to the work server.

**WSL2 notes**

- Keep a WSL terminal window open (minimised is fine) and turn off Windows sleep, or WSL stops and the miner with it.
- `nvidia-smi` inside WSL can read clocks and power but cannot set them. Overclock from Windows (MSI Afterburner or a similar tool).
- If the card also drives your display, desktop use can lag while mining. Stop the miner with `Ctrl+C`, or force a smaller batch such as `--batch 768`.

---

## Console

**Launch**: the banner, the setup summary, then one progress bar per GPU while kernels and batch sizes are being tested:

```
  ╔═════════════════╗
  ║ 18          Ar  ║   D A Z  ·  A R G O N
  ║                 ║   ══════════════════════════════════
  ║   ▄▀▀▀▄         ║   Argon2id GPU miner  ·  solo
  ║   █▄▄▄█ █▀▀▀    ║   v0.6.0  ·  RTX 30 / 40 / 50
  ║   █   █ █       ║
  ║                 ║   github.com/djdaz31/DazArgonMiner
  ║     39.948      ║
  ╚═════════════════╝
  ┌ Address   arg1q…
  ├ Work      explorer.argonmine.com  (solo)
  ├ Dev fee   1 %  (60 s / 100 min)
  ├ GPUs      4× RTX 3070 sm_86 · 1× RTX 4070 sm_89
  └ Kernel    fast2 · fast · original  (best per GPU, saved in tune.json, self-tested)

  GPU 0  ⠼  fast2 batch  768  Test 1/4 [━━━━━━━━────────────]  40%  25.99 kH/s
```

**Then the dashboard**, refreshed in place every 2 s:

```
╔═════════╗  DazArgonMiner 0.6.0 · Argon2id · solo                       uptime 02h14
║ 18   Ar ║  arg1q56234qn7f…shvc7tz                                     height 3 412
║  39.948 ║  fee 1 % · kernel fast/fast2 · 3 GPUs
╚═════════╝
─────────────────────────────────────────────────────────────────────────────────────
 GPU  model         hashrate     W   H/W   temp  fan    mem   krnl  batch  blk
   0  RTX 3070    23.60 kH/s   149   158    48°  50%   8001  fast2    768    0
   1  RTX 4070    28.30 kH/s   147   192    42°  63%  10501   fast   1024    1
   2  RTX 5080    53.24 kH/s   262   203    68°  30%  14801  fast2   1280    0
 ───  total       105.1 kH/s   558   188                                     1
─────────────────────────────────────────────────────────────────────────────────────
 » next block   ≈ 10 h 50 min       24 h chance    89 %
 » network      ≈ 15.18 MH/s        your share     0.69 %
 » this round   02h14  ▓▓▓░░░░░░░░░░░░░  21 % of expected
─────────────────────────────────────────────────────────────────────────────────────
 Blocks  1 total ≈ 49.5 ARG  A1 R0    last: #3 398 · GPU 1 · 11:42 · luck 0.42× · 14/100 conf
─────────────────────────────────────────────────────────────────────────────────────
 11:42  ✔ BLOCK 3 398 found by GPU 1   +49.5 ARG   luck 0.42×
 12:01  ⚠ GPU 2 at 80°
```

- **next block** = expected hashes for the current difficulty (from the work server's target) ÷ your hashrate. It is an average: in solo mining waiting twice as long happens often.
- **24 h chance** = probability of finding at least one block in the next 24 h at this hashrate.
- **network** is estimated from the difficulty and how fast the chain tip moves.
- **this round** = time since your last block (or since start) compared with the expected time. **luck** < 1 means a block came sooner than expected.
- Temperatures turn yellow at 70° and red at 80°, and a GPU whose miner restarts turns red. The frame flashes green when a block is found.
- A window narrower than 80 columns gets a compact table. Without a terminal (`--no-dashboard`, pipes, `NO_COLOR`), the miner prints plain lines instead: `[ RIG ]`, `[ ETA ]`, `[ NET ]` and `[ GPU ]` every minute, plus events.
- **HiveOS**: `miner` shows the dashboard, and `miner log` gets the plain lines.

A *stale* block means someone else found that height first. Repeated real rejects usually mean the memory overclock is too high.

The miner counts the blocks the network accepted when you submitted them. A block can still be orphaned later, so the [explorer](https://explorer.argonmine.com) is the reference for your balance.


---

## Options

`argonmine_hive.py`:

| Option | Default | |
|---|---|---|
| `--address` | *(required)* | payout address `arg1...` |
| `--api` | `https://explorer.argonmine.com` | work server |
| `--devices` | `all` | GPU indices, for example `0,1,3` |
| `--batch` | `auto` | `auto` = tuned per GPU, or one number for all GPUs |
| `--retune` | off | test kernels and batch sizes again |
| `--tune-seconds` | `45` | seconds per batch size while tuning |
| `--tune-file` | `<miner dir>/tune.json` | saved kernel + batch per GPU |
| `--blocks` | `<miner dir>/blocks.json` | block history |
| `--jobfile` | `/run/argonmine_job.txt` | current job, shared by the GPU processes |
| `--stats` | `/run/argonmine_stats.json` | stats for HiveOS |
| `--interval` | `2` | seconds between chain-tip checks |
| `--refetch` | `60` | refresh the job at least this often (jobs expire after 2 min) |
| `--summary` | `60` | seconds between plain summaries (plain mode and `--log`) |
| `--no-dashboard` | off | plain scrolling lines instead of the dashboard |
| `--log` | *(none)* | also write plain lines to this file |
| `--no-color` | off | no colours (implies plain lines) |

Environment: `ARGON_KERNEL=fast2|fast|orig` forces one kernel on every GPU.

`bin/argon_gpu --selftest` runs the three kernels against the known-answer tag and uses 8 MiB of VRAM.

---

## Performance

All figures were measured on real cards with this miner. 30 s runs are noisy, so treat a few % as noise.

> **fast2** is new in 0.6.0 and not in these tables yet. Every GPU measures fast2 against fast during tuning: the `rates` saved in `tune.json` show both figures for your cards.

**Kernel comparison on an RTX 5080** (same overclock, H/s):

| batch | fast kernel | original kernel |
|---:|---:|---:|
| 768 | 48,754 | 47,524 |
| 1024 | 46,646 | 34,207 |
| **1280** | **50,860** | 39,060 |
| 1536 | 50,186 | 42,986 |

**Tuned cards**:

| GPU | H/s | Settings |
|---|---:|---|
| RTX 5080 (WSL2) | ~53,200 | core +50, mem +1300, power limit 270 W, batch 1280 |
| RTX 4070 | ~28,300 | core lock 2700, mem offset +800, PL 180 W |
| RTX 3070 | ~23,600 | core +200, mem offset +1000, PL 150 W (power limited) |

The network grows quickly, so check the current hashrate on the explorer before estimating earnings.

---

## Overclocking

Argon2id with 8 MiB per hash is **memory-bound**: memory bandwidth matters most, the core much less.

- **Use a memory *offset*, not a memory *lock*.** HiveOS has both fields. A memory lock such as `810` or `1400` is common in profiles for other algorithms and pins the memory to a low clock. On a rig of four RTX 3070s and one RTX 4070, clearing the lock and setting offsets took it from 82 to 123 kH/s.
- Raise the memory offset in steps of +100 (+250 on GDDR7) and wait a few minutes at each step. **Keep the step that gives the most H/s, not the highest one that stays up.** GDDR6X and GDDR7 correct errors silently, so the hashrate drops before anything crashes.
- Then try lowering the core. If the H/s stay the same, keep the lower core: it saves power and heat.
- After changing the OC, restart once with `retune=1`. The `[selftest] fast kernel: OK` line also confirms the card still computes correct hashes at those settings.
- A card that draws exactly its power limit is power-limited. Raising PL helps if you can afford it.

HiveOS supports one OC profile per algorithm, so your `argon2id` profile stays separate from your other coins.

---

## Dev fee

**1 % of mining time**: 60 s every 100 minutes is mined to the developer's address. The fee is shown at start-up, and `[dev fee]` appears in the console while it is active. Blocks found during that window are announced as dev-fee blocks and are not added to your history. If you mine to the developer's address, there is no fee.

The fee is set by `DEV_FEE_PCT` and `DEV_ADDRESS` at the top of `argonmine_hive.py`. Leaving it on supports further work on the kernel.

---

## How it works

```
argonmine_hive.py ──GET /api/work──▶ explorer.argonmine.com
      │  writes job file (job id, 112-byte header prefix, target)
      ├──▶ argon_gpu  (GPU 0, nonce range 1) ─┐
      ├──▶ argon_gpu  (GPU 1, nonce range 2) ─┼─ HIT job nonce ──▶ POST /api/work/submit
      └──▶ argon_gpu  (GPU n, nonce range n) ─┘
```

- `argon_gpu` stays resident and reloads the job file when it changes, so a new block costs a file rewrite, not a process restart.
- **Fast kernel**: one 32-thread warp per hash and 4 hashes per CUDA block. The previous block stays in registers instead of being read back from VRAM. Warp-level sync only, with no `__syncthreads`. The BLAKE2b-based `P` permutation runs over a 1 KiB shared-memory tile per hash.
- **fast2 kernel**: same warp-per-hash layout, but each lane keeps 4 qwords of one row in registers. The row and column rounds exchange data with `__shfl_sync`, and shared memory is only used for the two row↔column transposes (padded against bank conflicts). Reference blocks are read with `ld.global.cg`. In slices 0–1 (data-independent addressing) the next reference block is loaded while the current block is computed.
- **Correctness**: `tools/argon2ref.py` is a pure-Python Argon2id that reproduces the RFC 9106 test vector. It generated the known-answer tag that the CUDA kernels check against at every start. The CUDA source of the three kernels was also compiled and run on CPU threads emulating the warp, and all of them matched that tag and the Python reference on other inputs.
- The build applies two small patches (`patch_lowcpu.py`, `patch_fast.py`) to the original source in `src/argon_gpu.cu`. If the fast kernel fails to compile, the original kernel is built instead.

**Work-server limits** (from the Argonmine docs): jobs expire 2 minutes after issue, at most 8 open jobs per payout address, and at most 3 solutions checked per job.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `tar: not in gzip format` | You downloaded a web page. Use the **release asset** link (`/releases/download/...`) or `raw.githubusercontent.com`, not a `/blob/` link |
| `[config] '' does not look like an ARG address` | Set the flight-sheet wallet to your `arg1...` address and the template to `%WAL%` |
| `[run] build failed` | Send the `[fetch]`/`[build]` lines from `miner log`. The rig needs to reach `developer.download.nvidia.com`, or have `nvcc` 12.8+ installed |
| `selftest ... MISMATCH` | The card computes wrong hashes at the current OC. Lower the memory offset. The miner falls back to the original kernel, or refuses to mine if both kernels fail |
| `batch N: failed (VRAM?)` while tuning | Normal: that batch does not fit, and tuning skips it |
| One CPU core at 100 % per GPU | You are running a build older than 0.1.4. Update |
| Hashrate much lower than expected | Check for a memory **lock** in the OC profile (see [Overclocking](#overclocking)) |
| `[work] rate-limited` | Too many machines behind one IP. Add `--interval 4` on the extra ones |

---

## Changelog

- **0.6.0**: dashboard (time to next block, 24 h chance, network estimate, round, blocks, events); periodic-table launch banner; new fast2 kernel; kernel and batch tuned per GPU; `kernel=` option.
- **0.5.0**: start-up banner (address, server, fee, algorithm, GPU architectures, kernel list); animated progress bars while tuning; HiveOS log written by the miner itself.
- **0.4.1**: prebuilt binary in both archives (HiveOS and `linux`), nothing compiled at deploy time; automatic rebuild if the binary cannot run; `run.sh` launcher for Ubuntu/WSL2.
- **0.4.0**: batch tuned per GPU and saved in `tune.json`; 1 % time-based dev fee; job refresh every 60 s; stale and reject reasons shown.
- **0.3.0**: GPU summary table, block-found banner and block history (`blocks.json`).
- **0.2.0**: fast warp-per-hash kernel, start-up self-test with automatic fallback.
- **0.1.x**: HiveOS package, multi-GPU driver, per-GPU nonce ranges, CUDA toolchain fetched without pip, low-CPU waiting.

---

## Credits and license

- CUDA Argon2id implementation: [YGYxop6/argonmine-cuda-miner](https://github.com/YGYxop6/argonmine-cuda-miner), MIT License, © 2026 YGYxop6. Included unmodified in `src/argon_gpu.cu` apart from the `--nonce-seed` option, with its license in `LICENSE.argon_gpu`.
- Fast kernel, HiveOS package, multi-GPU driver, tuning and console: this repository, MIT License (see `LICENSE`).

**Disclaimer**: this is an unofficial miner, not affiliated with the Argonmine project. It is experimental software provided as is. Mining wears hardware, and overclocking can make cards unstable. Argonmine is a young network: its coin has no market outside the project's own swap service, and its bridge holds funds without an external audit. Mine at your own risk.
