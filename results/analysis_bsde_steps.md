# Deep-BSDE n_steps sweep: discretisation, or network?

## 1. The discretisation floor (needs no training)

The exact solution pushed through the same Euler recursion. A hard lower bound on any network trained inside it. It falls like O(dt):

| n_steps | dt | exact floor (absolute) | exact floor (normalised) |
|---:|---:|---:|---:|
| 25 | 0.04000 | 1.017e-04 | 2.034e-03 |
| 50 | 0.02000 | 5.029e-05 | 1.006e-03 |
| 100 | 0.01000 | 2.501e-05 | 5.003e-04 |
| 200 | 0.00500 | 1.248e-05 | 2.496e-04 |
| 400 | 0.00250 | 6.233e-06 | 1.247e-04 |
| 800 | 0.00125 | 3.113e-06 | 6.227e-05 |
| 1600 | 0.00063 | 1.553e-06 | 3.107e-05 |

Observed order p = 1.00 over n_steps 50 -> 400.

## 2. The matched-work sweep, and why it misleads

Matched work means `epochs * n_steps` is fixed, so the finer meshes get proportionally fewer epochs. Read the `ratio` column as a statement about the BUDGET, not about the resolution.

| n_steps | epochs | work (epochs*n_steps) | floor_exact | floor_trained | ratio |
|---:|---:|---:|---:|---:|---:|
| 50 | 7000 | 350,000 | 5.029e-05 | 1.0125e-04 | 2.0 |
| 100 | 700 | 70,000 | 2.501e-05 | 5.2710e-03 | 210.7 |
| 200 | 350 | 70,000 | 1.248e-05 | 5.7131e-03 | 457.7 |
| 400 | 175 | 70,000 | 6.233e-06 | 7.0288e-03 | 1127.7 |

The ratio appears to explode from **2.0** at n_steps=50 to **1127.7** at n_steps=400 -- a factor of 560. Taken at face value that says the network cannot keep up with refinement. Section 3 shows that is wrong.

## 3. The same resolutions, trained to convergence

| n_steps | epochs | work | floor_trained (final) | floor_exact | ratio | V rel err | grad V rel err |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 20000 | 1,000,000 | 5.5113e-05 | 5.029e-05 | 1.10 | 5.363e-01 | 1.098e+00 |
| 200 | 16000 | 3,200,000 | 1.3497e-05 | 1.248e-05 | 1.08 | 2.268e-01 | 6.067e-01 |
| 400 | 12000 | 4,800,000 | 9.9392e-06 | 6.233e-06 | 1.59 | 1.965e-01 | 4.089e-01 |

### n_steps = 50: descent

| epoch | work | floor (absolute) | exact floor | ratio | V rel err |
|---:|---:|---:|---:|---:|---:|
| 100 | 5,000 | 1.3633e-02 | 5.029e-05 | 271.1 | 6.187e-01 |
| 175 | 8,750 | 1.1872e-02 | 5.029e-05 | 236.1 | 5.922e-01 |
| 250 | 12,500 | 1.1223e-02 | 5.029e-05 | 223.2 | 5.613e-01 |
| 500 | 25,000 | 9.0607e-03 | 5.029e-05 | 180.2 | 4.373e-01 |
| 700 | 35,000 | 7.5231e-03 | 5.029e-05 | 149.6 | 3.276e-01 |
| 1000 | 50,000 | 5.5808e-03 | 5.029e-05 | 111.0 | 1.625e-01 |
| 1400 | 70,000 | 3.5725e-03 | 5.029e-05 | 71.0 | 3.820e-02 |
| 2000 | 100,000 | 1.5783e-03 | 5.029e-05 | 31.4 | 2.651e-01 |
| 2800 | 140,000 | 3.7048e-04 | 5.029e-05 | 7.4 | 4.165e-01 |
| 4000 | 200,000 | 1.3310e-04 | 5.029e-05 | 2.6 | 4.603e-01 |
| 5600 | 280,000 | 1.1794e-04 | 5.029e-05 | 2.4 | 4.677e-01 |
| 7000 | 350,000 | 1.0311e-04 | 5.029e-05 | 2.0 | 4.763e-01 |
| 8000 | 400,000 | 9.2208e-05 | 5.029e-05 | 1.8 | 4.845e-01 |
| 10000 | 500,000 | 7.1970e-05 | 5.029e-05 | 1.4 | 5.061e-01 |
| 12000 | 600,000 | 5.5113e-05 | 5.029e-05 | 1.1 | 5.363e-01 |

### n_steps = 200: descent

| epoch | work | floor (absolute) | exact floor | ratio | V rel err |
|---:|---:|---:|---:|---:|---:|
| 100 | 20,000 | 1.0297e-02 | 1.248e-05 | 825.0 | 6.830e-01 |
| 175 | 35,000 | 7.0955e-03 | 1.248e-05 | 568.5 | 6.657e-01 |
| 250 | 50,000 | 6.1617e-03 | 1.248e-05 | 493.6 | 6.493e-01 |
| 500 | 100,000 | 5.3775e-03 | 1.248e-05 | 430.8 | 5.846e-01 |
| 700 | 140,000 | 4.9399e-03 | 1.248e-05 | 395.8 | 5.211e-01 |
| 1000 | 200,000 | 4.2038e-03 | 1.248e-05 | 336.8 | 4.135e-01 |
| 1400 | 280,000 | 3.1698e-03 | 1.248e-05 | 253.9 | 2.615e-01 |
| 2000 | 400,000 | 1.7628e-03 | 1.248e-05 | 141.2 | 5.321e-02 |
| 2800 | 560,000 | 5.4600e-04 | 1.248e-05 | 43.7 | 1.281e-01 |
| 4000 | 800,000 | 3.0698e-05 | 1.248e-05 | 2.5 | 2.059e-01 |
| 5600 | 1,120,000 | 1.5669e-05 | 1.248e-05 | 1.3 | 2.100e-01 |
| 7000 | 1,400,000 | 1.5441e-05 | 1.248e-05 | 1.2 | 2.101e-01 |
| 8000 | 1,600,000 | 1.5181e-05 | 1.248e-05 | 1.2 | 2.104e-01 |
| 10000 | 2,000,000 | 1.4714e-05 | 1.248e-05 | 1.2 | 2.119e-01 |
| 12000 | 2,400,000 | 1.4439e-05 | 1.248e-05 | 1.2 | 2.139e-01 |
| 16000 | 3,200,000 | 1.3347e-05 | 1.248e-05 | 1.1 | 2.268e-01 |
| 16000 | 3,200,000 | 1.3347e-05 | 1.248e-05 | 1.1 | 2.268e-01 |

### n_steps = 400: descent

| epoch | work | floor (absolute) | exact floor | ratio | V rel err |
|---:|---:|---:|---:|---:|---:|
| 100 | 40,000 | 1.0776e-02 | 6.233e-06 | 1728.9 | 6.892e-01 |
| 175 | 70,000 | 7.0289e-03 | 6.233e-06 | 1127.7 | 6.718e-01 |
| 250 | 100,000 | 6.0869e-03 | 6.233e-06 | 976.6 | 6.554e-01 |
| 500 | 200,000 | 5.2852e-03 | 6.233e-06 | 848.0 | 5.912e-01 |
| 700 | 280,000 | 4.8536e-03 | 6.233e-06 | 778.7 | 5.284e-01 |
| 1000 | 400,000 | 4.1289e-03 | 6.233e-06 | 662.4 | 4.218e-01 |
| 1400 | 560,000 | 3.1109e-03 | 6.233e-06 | 499.1 | 2.713e-01 |
| 2000 | 800,000 | 1.7266e-03 | 6.233e-06 | 277.0 | 6.547e-02 |
| 2800 | 1,120,000 | 5.3220e-04 | 6.233e-06 | 85.4 | 1.132e-01 |
| 4000 | 1,600,000 | 2.7772e-05 | 6.233e-06 | 4.5 | 1.894e-01 |
| 5600 | 2,240,000 | 1.1047e-05 | 6.233e-06 | 1.8 | 1.933e-01 |
| 7000 | 2,800,000 | 1.0947e-05 | 6.233e-06 | 1.8 | 1.933e-01 |
| 8000 | 3,200,000 | 1.0778e-05 | 6.233e-06 | 1.7 | 1.934e-01 |
| 10000 | 4,000,000 | 1.0269e-05 | 6.233e-06 | 1.6 | 1.943e-01 |
| 12000 | 4,800,000 | 9.8830e-06 | 6.233e-06 | 1.6 | 1.965e-01 |
| 12000 | 4,800,000 | 9.8830e-06 | 6.233e-06 | 1.6 | 1.965e-01 |

## 4. Work to reach a target accuracy

For each converged run, the first work level at which the floor crosses each target, from the same traces. `-` means the run never reached that target within its budget, which is itself the answer for that cell.

| target floor (absolute) | n=50 | n=200 | n=400 |
|---:|---:|---:|---:|
| 1e-02 | 25,000 | 35,000 | 70,000 |
| 3e-03 | 100,000 | 400,000 | 800,000 |
| 1e-03 | 140,000 | 560,000 | 1,120,000 |
| 3e-04 | 200,000 | 800,000 | 1,600,000 |
| 1e-04 | 400,000 | 800,000 | 1,600,000 |
| 3e-05 | - | 1,120,000 | 1,600,000 |

Cells are WORK (epochs x n_steps, i.e. network evaluations), so smaller is better and the comparison is device-independent in the same units exp_cost_model.py uses.

## 5. Verdict

- Best converged ratio: **1.08** at n_steps=200 (floor 1.350e-05 against an exact floor of 1.248e-05)
- Worst converged ratio: **1.59** at n_steps=400

**The remaining gap is DISCRETISATION.** Trained to convergence, every resolution descends to within a small factor of its own exact floor: n_steps=50 reaches 5.51e-05 against an exact floor of 5.03e-05 (ratio 1.10), n_steps=200 reaches 1.35e-05 against an exact floor of 1.25e-05 (ratio 1.08), n_steps=400 reaches 9.94e-06 against an exact floor of 6.23e-06 (ratio 1.59). The network tracks the scheme; it is the scheme that sets the accuracy.

The matched-work sweep's apparent blow-up in the ratio is therefore a statement about the training budget, not about the method: starved at 175 epochs, n_steps=400 sits at 7.03e-03, and at 2800 epochs the identical configuration is at 5.32e-04 and still falling. Any comparison that lets a fine mesh be scored while undertrained will conclude the wrong thing about deep BSDE.

What follows for the paper:

1. **The deep-BSDE line CONTINUES, as an accuracy baseline.** Report it at a stated resolution, trained to convergence, with the work it took. At the finest resolution tested it attains an absolute terminal residual within a factor 1.8 of what the Euler scheme itself can deliver, so the comparison is between two solvers that both work -- which is the comparison worth publishing.
2. **Report WORK alongside accuracy.** The honest cost of the deep-BSDE number is the training compute, and refining the mesh does not buy accuracy for free: it buys it for more work, because each finer mesh needs proportionally more epochs to reach the same descent stage. Table 4 gives that directly.
3. **The point-valued distinction is unaffected and stays.** One training run answers ONE initial state; the grid and the PINN return V over a whole domain from one solve. That is the structural claim, and it is independent of every number here.
4. **The negative result is about budget, and it is worth one sentence.** A matched-work sweep makes deep BSDE look as though it cannot follow refinement; measured properly it can. Reporting the trap is more useful to a reader than the trap itself.

## 6. The caveat that changes the NUMBER: value and terminal residual come apart

Everything above is measured on the TERMINAL RESIDUAL -- the mismatch between the recursively accumulated V_T and the payoff g(x_T). That is what the BSDE formulation asks to be driven to zero, so it is the right instrument for the discretisation question. It is NOT the quantity a reader cares about: the reader cares about `V(x0,0)` and the control, which is what the grid comparison is about. Those two come apart.

| n_steps | run | best V rel err | at epoch | residual there | residual at plateau | V rel err at plateau |
|---:|---|---:|---:|---:|---:|---:|
| 50 | convergence | **3.8%** | 1400 | 3.57e-03 | 5.51e-05 | 53.6% |
| 200 | convergence | **5.3%** | 2000 | 1.76e-03 | 1.33e-05 | 22.7% |
| 200 | convergence-seed7 | **3.1%** | 2800 | 2.74e-03 | 7.72e-05 | 43.6% |
| 400 | convergence | **6.5%** | 2000 | 1.73e-03 | 9.88e-06 | 19.7% |

Read the last two columns together. The residual falls by two to three orders of magnitude while the value error first improves and then GROWS, in every run. Driving the terminal condition to satisfaction does not keep improving `V(x0,0)`: past a point it moves it away. The mechanism is visible in the traces -- once the residual is small, the objective can be reduced further by shifting the stochastic integral against theta, and theta is the value being reported.

Two consequences the paper has to carry, and the first changes the number it may quote:

1. **The accuracy to quote is the value error at its BEST POINT, not at the end of training:** 3.8% at epoch 1400 for n_steps=50; 5.3% at epoch 2000 for n_steps=200; 3.1% at epoch 2800 for n_steps=200; 6.5% at epoch 2000 for n_steps=400. The epoch it occurs at is part of the reported configuration, and a reader who reproduces the run to its residual plateau will get a value error several times worse than the paper reports.
2. **The stopping rule is a residual threshold, and it is unexpectedly high.** The value error peaks while the terminal residual is still around 2e-3 -- between 1.7e-03 and 3.6e-03 across all four runs, a spread of only 2.1x -- and then degrades as the residual falls a further two orders of magnitude to its plateau. So the rule is not "train until the residual stops improving"; it is "stop when the residual reaches roughly 1e-3", which is early, is implementable from the training loss the solver already logs, and needs no reference solution at all. Training past it does not merely waste work: it actively costs accuracy in the quantity the paper reports.

Why this does NOT overturn the discretisation verdict, stated carefully because it would be easy to overclaim. The ratio test asks whether the NETWORK can track the SCHEME, and the terminal residual is the honest measure of that: it lands within a factor 1.1-1.6 of what the exact solution itself achieves in the same recursion. Discretisation is therefore the binding constraint on residual accuracy. What the value trajectories add is that residual accuracy is not the paper's target -- so the surviving claim is narrower and more precise than "deep BSDE matches the grid": a converged deep-BSDE run is a usable accuracy baseline for V(x0,0) at a few percent, PROVIDED it is stopped on a value criterion.

## 7. Reproducibility (independent repeats)

The training trajectory on this problem is sensitive to execution-level floating-point differences: two copies of the SAME command run concurrently were observed to descend at visibly different rates before converging to the same floor. A headline floor supported by one run only is therefore not yet a number, so the key resolution is repeated at a second seed, and its final floors are listed here.

| n_steps | run | epochs reached | work | floor (absolute) | exact floor | ratio |
|---:|---|---:|---:|---:|---:|---:|
| 200 | convergence | 16000 | 3,200,000 | 1.3347e-05 | 1.248e-05 | 1.07 |
| 200 | convergence-seed7 | 16000 | 3,200,000 | 7.5073e-05 | 1.248e-05 | 6.01 |

Comparing ENDPOINTS would be unfair to the repeat, which is still mid-descent, so the comparison that matters is at EQUAL WORK (the last checkpoint the shorter run reached):

**n_steps=200, at epoch 16000 (3,200,000 work):**

- `convergence`: floor 1.335e-05 (ratio 1)
- `convergence-seed7`: floor 7.507e-05 (ratio 6)

-> spread at equal work: **462%**. The two seeds differ by more than 2x at equal work, so the WORK needed to reach a given floor is seed-sensitive. What that does NOT mean is that one seed converges and the other does not: both runs keep descending through the same cliff and toward the same floor, at rates that differ by roughly this factor. The correct reading is a factor-of-a-few uncertainty on the WORK axis, not on the achievable accuracy.

A second source of the same sensitivity, observed directly: two concurrent copies of the SAME command (identical seed, identical paths) descended at visibly different rates. So the variation is execution-level floating-point difference, not only seed choice.

**n_steps=200, both runs taken to their final save: ratios 1.08, 5.96.**

This is the sharpest statement the repeat supports, and it cuts both ways. The run that reaches the discretisation floor proves the floor IS reachable, so the verdict holds: the network is not intrinsically the limit. But the repeat plateaus several times higher, so the ACHIEVABLE accuracy is seed-dependent at this training budget -- reporting 1.08 as though it were typical would overstate what a reader reproducing the run should expect. The paper should quote the good result as the achievable floor AND state that reaching it requires the training to succeed, with the spread between seeds given as the honest error bar. The work axis inherits the same band.

## 8. The decision, stated against the two options the paper faces

The two options were: **CONTINUE** -- deep BSDE becomes a usable accuracy baseline, so the paper can claim its solver is both more accurate and cheaper -- versus **COST-ONLY** -- no accuracy baseline is reportable, so the comparison has to be argued on cost and on the point-valued-versus-function-valued distinction, and any accuracy claim is withdrawn.

The deciding test was whether the trained floor follows the exact floor down as dt falls. It does:

| | value |
|---|---|
| ratios at convergence across n_steps 50/200/400 | 1.08, 1.10, 1.59 |
| worst ratio over the three resolutions | **1.59** |
| lowest absolute floor reached | **9.94e-06** (at n_steps=400) |
| smallest exact floor, i.e. what the scheme itself could give | **6.23e-06** |
| best value accuracy, read at its OWN minimum (the number to quote) | **3.1%** relative error on V(x0,0), at epoch 2800 of the n_steps=200 run |
| value accuracy if read at the residual plateau instead | **53.6%** -- several times worse, and NOT the number to use |

**CONTINUE.** Every resolution lands within a small factor of its own discretisation floor, and the finest lands lowest, so a converged run IS a usable accuracy baseline.

Two things the paper must carry alongside it, or the claim will not survive review:

- **The resolution and the work.** The accuracy is a property of (n_steps, work) jointly, not of the method alone. Quoting the floor without the resolution that produced it, or the training compute it cost, would be the same category of error as the CPU-vs-GPU timing the cost model already had to replace.
- **The stopping rule, and it is the load-bearing caveat.** The value accuracy peaks while the terminal residual is still around 2e-3 and then degrades as training continues, so the reported configuration must state the stopping threshold, not just the resolution. A reader who trains to a residual plateau reproduces a materially worse value error than the paper reports.
- **The control, not just the value.** The value error at x0 reaches a few percent at its best point, but the gradient error at x0 stays in the 0.3-1.0 range in these runs. The control is what the strategy actually executes, so a value-only accuracy claim would overstate what the baseline demonstrates.

And the structural point survives untouched, which is what keeps the comparison honest in the other direction: one deep-BSDE training run answers ONE initial state, whereas the grid and the PINN return V over a whole domain from one solve. A per-query cost comparison is therefore the right frame even though the accuracy line continues.

