# Training speed: 2.1 times as many steps per second

Training now runs at about **20,800 steps per second** on the sandbox machine, up from about **9,800** with the code
as it was for the 30-minute run. Steps per second here is the program's "Overall Steps/Second": timesteps collected,
divided by the time for collecting them plus the time for learning from them.

Part of this gain is specific to machines like this one, where torch uses fewer threads than the machine has.
Against the old code forced to use both CPU threads (13,300–13,400 steps per second), the new code is **1.55 times**
as fast. That is probably the better guide for a machine where torch already uses all cores. I could not measure one.

The training itself is unchanged. The updates are the same as before up to float rounding, and training from scratch
follows the same curves (both checked, see "Same results").

All numbers are from one machine: 2 CPU threads (Xeon 2.6 GHz with AVX-512), 1.9 GB RAM, no GPU.
**I could not measure a GPU**, and most of the gain is in learning on the CPU, so it will not carry over to a GPU as it is.

## Measurements

Settings of `train_cpu.sh` (64 games, 24,000 steps per iteration, batches of 6,000, 2 epochs), continuing from the
14.1M-step checkpoint. Each number is one run of 30 iterations after a warm-up iteration. Both programs were run twice,
taking turns.

| | Before | After | |
|---|---|---|---|
| **Overall steps per second** | 9,833 and 9,843 | 20,775 and 21,104 | **2.1x** |
| Collecting only (steps/s) | 51,900 and 52,700 | 56,500 and 56,900 | 1.08x |
| Learning only (steps/s) | 12,100 and 12,100 | 32,900 and 33,500 | 2.7x |
| Peak memory | 0.5 GB | 0.38 GB | |

Time per 1,000 steps, in milliseconds:

| Phase | Before | After |
|---|---|---|
| Game simulation, observations, rewards | 8.7 – 9.0 | 8.8 – 8.9 |
| Choosing actions (policy network) | 8.7 | 7.1 – 7.3 |
| Other collection work | 1.6 | 1.5 – 1.6 |
| Value prediction and GAE | 6.3 – 6.5 | 2.2 – 2.3 |
| PPO update | 76.0 – 76.3 | 27.5 – 28.2 |

Three more runs of the final code gave 20,126, 20,834 and 21,051 steps per second.

More measurements, one run each:

- **The old code with both CPU threads.** Out of the box the old code learns on one thread here (see change 5).
  `OMP_NUM_THREADS=2` alone does not change that (9,917). With `MKL_DYNAMIC=FALSE` added, it learns on two threads
  and reaches 13,262 (13,439 with `OMP_WAIT_POLICY=passive`). Learning gets faster (41–47 ms per
  1,000 steps instead of 76), but collecting gets slower (31,500–41,800 steps per second instead of 52,000),
  apparently because torch's threads then compete with the simulation threads.
- **Training from scratch** for 4 million steps with the new code: 21,670 steps per second over the whole run.
- **The original example's settings on the CPU** (256 games, 50,000 steps per iteration in one batch, 1 epoch):
  11,491 → 29,183 steps per second (2.5x), and peak memory 1.8 GB → 0.74 GB.

Single runs of 10 iterations differ by about ±3% from each other, so differences of a few percent below are uncertain.
All 60 benchmark runs of this work are in `results/speed/benchmarks.jsonl`, in the order they were measured.

## What was slow, and what changed

A profile of the old code showed where the time went: 51% in matrix multiplications that ran on one thread, 15% in
layer normalization, ReLU and copying, and 19–25% in the game simulation.

| State of the code | Steps per second |
|---|---|
| Starting point | 9,700 – 10,500 |
| With changes 1–4 below, still learning on one thread | 13,300 – 14,000 |
| With change 5: learning on both CPU threads | 19,700 – 21,400 |
| With changes 6–7: collection loop, compiler options | 20,800 – 21,700 |

1. **The shared layers were computed twice per update**, once for the policy and once for the critic, forward and
   backward. They are now computed once and feed both.
2. **Linear → LayerNorm → ReLU is one operation.** Torch makes a separate pass over memory for each of them and for
   each of their backward steps. `FusedLayer.cpp` does everything except the matrix multiplication row by row in one
   forward and one backward kernel, vectorized for AVX-512 and AVX2. Without it: 12,000 instead of 13,300–14,000.
3. **Minibatches that fit the CPU cache.** A batch is sent through the network in pieces of 1,024 samples instead of
   6,000 (or 50,000) at once. The gradients of the pieces are added up before the optimizer step, so the update is the
   same. This is also what lowers the memory use.
4. **Freed memory is kept.** glibc gave the memory of every freed tensor back to the system and requested it again
   for the next one. That caused 130,000 page faults per iteration, now about 150. Changes 3 and 4 both contribute
   to this. Without change 4 alone the speed was 13,500, inside the range with it, so on its own it adds little.
5. **Learning uses all CPU threads.** Torch chooses its thread count from the number of physical cores. This virtual
   machine reports its 2 CPUs as one core with 2 hyperthreads, so torch used 1. The learner now measures at startup
   whether more threads help (here: 1.94x for matrix products), and each thread computes the gradients of its own
   minibatches. The value prediction before GAE is split the same way.
6. **Collection loop.** The next simulation step starts before the observations are checked and stored, and
   trajectories are stored without temporary copies. No measurable gain on this machine.
7. **Compiler options** `-fno-semantic-interposition -fno-math-errno` and linker option `-Bsymbolic-functions` (in
   `CMakeLists.txt`). In five pairs of runs taking turns, collecting was 2–10% faster with them and the total was
   0–10% faster, 5% on average.

Tried and dropped: letting torch itself use two threads (18,300–18,500, and unstable, because its waiting threads
compete with the simulation threads) and in-place ReLU (no gain).

## Same results

- **Math test:** `build/GigaLearnTests` computes the gradients of a batch with the original formulation and with the
  new code, and compares both to the same computation in 64-bit floats. With the trained checkpoint, the new code
  differs from the 64-bit result by 1.4e-07 (relative, over all gradients together) and the original code by 6.8e-07.
  Two threads against one: 6e-08. Network outputs with fused layers against torch's layers: 3.5e-07. Value predictions
  in parts against all at once: 3.2e-07.
- **Training from scratch**, 4 million steps, same seed and settings as the 30-minute run. The runs are not identical
  step by step (different rounding, and the games are simulated on several threads), but the curves agree:

  | Average over steps 3M–4M | 30-minute run (old code) | New code |
  |---|---|---|
  | Reward per step | 0.903 | 0.916 |
  | Policy entropy | 0.7334 | 0.7322 |
  | KL divergence per update | 0.0078 | 0.0078 |
  | Time in the air | 83.7% | 83.4% |

  The metrics of this run are in `results/speed/from_scratch_4m_metrics.jsonl`.
- **Checkpoints** from before load and continue as before.
- **Evaluation** of the 14.1M checkpoint against random actions: 6.51 touches per minute with the old code
  (120 minutes), and with the new code 6.37 (720 minutes) and 6.1 to 6.7 (five runs of 120 minutes).

What does behave differently:

- `--minibatch-size` (`miniBatchSize`) no longer has to divide the batch size. On a CPU its default is now about
  1,024 (the batch is split into equal pieces, the same number for every thread) instead of the whole batch. On a
  GPU the default is still the whole batch.
- Reported averages (entropy, KL divergence, losses) are weighted by the number of samples in each minibatch.
- With a guiding policy (not used by the example), its loss is now weighted by each minibatch's share of the batch.
  Before, it counted once per minibatch, so it grew with the number of minibatches.
- The compiler options can change the simulation's float rounding in the last digit. I did not check whether the
  simulation is bit-identical to the earlier build.

One thing I found and left alone: with `overbatching` (on by default), the samples left over after dividing an
iteration into batches are added to the last batch, but the update then only uses the first `batchSize` samples of it.
At these settings about 3% of the samples are skipped in each epoch (838 of 24,838 per iteration on average).
Using them would be a change to the training, not a speed-up: each update would take about 3% longer, which lowers
steps per second by about 2%.

## What limits it now

Both CPU threads are busy 93% of the time. 53% of the CPU time is matrix multiplication in MKL, which by my estimate
runs at 85–90% of what these two cores can do, and 23% is the game simulation. What is left is spread thinly:

- The policy loss (softmax, entropy, clipping) takes about 5% of the update as separate torch operations. A fused
  kernel like the one for the layers would recover perhaps 2–3% overall. Not done.
- Link-time or profile-guided optimization of the simulation: a few percent of the simulation time. Not done.
- The games are handed to the thread pool one by one. On this machine the pool's own code is 0.5% of the collection
  time. With many cores and hundreds of games it may matter more. Not measured.

Settings (not used for any number above):

- `--epochs 1`, the original example's value: 28,200 steps per second. This changes the training: every step is
  learned from once instead of twice.
- `--games 128` or `256` instead of 64: no clear difference in the rate at which steps reach the learner
  (18,800–20,300 against 19,200–20,800). The overall number is higher in the first dozens of iterations of a run
  (up to 24,600), because steps of unfinished episodes pile up before the learner gets them.

Later, for the bigger networks of the 1v1/2v2/3v3 bot, two more things were added (see `ARCHITECTURE.md`): actions
are inferred on all CPU threads when the policy is big, and every learner thread gets the same number of minibatches.
Neither changes the example bot's speed: measured in turns on that day, 16,621 and 16,308 with the newer code against
16,294 and 15,917 with the code of this report. The machine itself was about 20% slower that day than for the numbers
above, so expect the benchmark to vary by that much between sessions.

## Not tested

- **GPU.** On a GPU the changes that apply are the single pass through the shared layers, the collection loop and the
  compiler options. The fused layers and the learner threads are only used on the CPU. I had no GPU, so the GPU path
  of the changed code has not been run.
- **Windows.** The fused layers need GCC or Clang. With MSVC the regular torch layers are used. I did not build there.
- **Machines with more cores.** The thread count is chosen automatically, and `--threads N` overrides it.
  I could only measure 2 threads.

## How to check

```bash
./setup_linux.sh                                  # builds build/GigaLearnBot
python3 tools/benchmark.py --iterations 30        # steps per second and time per phase, settings of train_cpu.sh
python3 tools/benchmark.py -- --epochs 1          # the same with other GigaLearnBot options
python3 tools/benchmark.py --no-preset -- --device cpu     # the original example's settings

cmake --build build --target GigaLearnTests       # the math test (not built by default)
./build/GigaLearnTests checkpoints/14102144
```

The benchmark copies the checkpoint to a temporary folder and does not change `checkpoints/`.
`GGL_NO_FUSED_LAYERS=1` and `GGL_NO_MALLOC_TUNING=1` switch off changes 2 and 4, `--threads 1` switches off change 5.

## Files

| File | Change |
|---|---|
| `GigaLearnCPP/src/private/GigaLearnCPP/Util/FusedLayer.h/.cpp` | New: the fused layer kernels with their backward pass |
| `GigaLearnCPP/src/private/GigaLearnCPP/Util/Models.h/.cpp` | Use the fused layers on the CPU |
| `GigaLearnCPP/src/private/GigaLearnCPP/PPO/PPOLearner.h/.cpp` | Shared layers once, minibatches, learner threads, value prediction in parts |
| `GigaLearnCPP/src/public/GigaLearnCPP/Learner.h/.cpp`, `LearnerConfig.h` | Thread counts, memory settings, collection loop |
| `GigaLearnCPP/tests/TestLearnerMath.cpp` | New: the math test |
| `CMakeLists.txt`, `GigaLearnCPP/CMakeLists.txt` | Compiler options, test target |
| `src/ExampleMain.cpp` | `--threads`, `--minibatch-size` |
| `tools/benchmark.py` | New: the benchmark |
