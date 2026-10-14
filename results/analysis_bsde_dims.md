# Does deep BSDE reach a useful value at higher dimension?

Deep BSDE is the standard high-dimensional method in mathematical finance, so a comparison paper that cites it without running it leaves its most likely referee question unanswered. These are the runs that answer it. Every number is a measured trace, not a fitted model.

| d | exact floor | residual at epoch 100 | best V error | at epoch | residual there | residual / start | residual / floor |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5.029e-05 | 1.363e-02 | **3.820%** | 1400 | 3.573e-03 | 26.20% | 71x |
| 4 | 2.012e-04 | 1.321e+02 | **3.274%** | 2000 | 5.212e+00 | 3.94% | 25907x |
| 8 | 4.023e-04 | 8.555e+01 | **0.348%** | 6000 | 7.170e-01 | 0.84% | 1782x |

## What the table says

**Deep BSDE reaches a usable value at every dimension measured.** The best value error is 0.35% to 3.82% across d = 1, 4, 8. So it can be reported as a baseline, and the paper's omission of it is a fixable gap rather than an impossibility.

**But only if training stops at the right point.** Read at the end of each run instead, the value error is 53.6% to 1267.6% -- worse than the best by a factor of 14.0 to 686. In every run the residual kept falling while the value degraded: the residual is a progress signal for the SCHEME, not for the value.

**The stopping threshold is problem-relative, and the absolute one does not transfer.** Expressed as a fraction of the residual at epoch 100, the optimum sits at 0.8% to 26.2% across dimensions -- a spread of 31x. Expressed as a multiple of the discretisation floor it sits at 71x to 25907x, a spread of 365x. So the scale-free framing is the right one, but it is not a tight rule: the fraction still moves by a factor of 31 across the dimensions measured, which is far better than the 365x of the absolute framing and still not something to quote as a constant.

Two consequences, and they point the same way.

The first is a warning for anyone reproducing this: the d=1 threshold of ~1e-3 applied literally at d=4 stops three orders of magnitude too late, well past the point where the value has started to diverge, so the reproduced number would be an order of magnitude worse than the one reported here for reasons that have nothing to do with the method.

The second is that neither framing is safe to hard-code, so the stopping point has to be found by monitoring the value rather than by a rule -- which is the same conclusion the d=1 study reached, now with the evidence that a rule cannot be substituted for it.

## Why this is not a rescue of deep BSDE as a competitor

At its best point the solver is still 71x to 25907x the discretisation floor of the scheme it is solving, and the residual is still falling at the end of every run. So deep BSDE does not reach the grid's accuracy where the grid can run -- which is the paper's existing claim, now supported against the method a referee would most likely name. What the dimension study adds is a measured statement about the regime where deep BSDE is usable and at what cost, which is what turns this from a missing baseline into a result.

