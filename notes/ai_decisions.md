# AI assistant decisions, mistakes and uncertainties

Ventilator pressure prediction, Nov 3–15, 2026. Written by the AI assistant (Claude) at the user's request.

**Latest result (round 3, Nov 15):** test MAE 0.18259, out-of-fold CV MAE 0.1992. Details in "Round 3" at the end.
**Round 2 (Nov 12):** test MAE 0.18935, OOF 0.2118.
**Round 1 (Nov 3):** test MAE 0.20138, OOF 0.2332. Median baseline CV MAE 6.07.

## Approach decisions

- **Model choice is not original.** I didn't look anything up, but I knew from background knowledge that bidirectional LSTMs with a cumulative `u_in` (volume) feature were a common approach for this competition. I said so in the plan, before writing code.
- **Features:** lags, leads and differences of `u_in`; cumulative `u_in·dt` as a volume proxy; R·`u_in` and volume/C, motivated by P ≈ P₀ + R·flow + V/C; per-breath max and mean of `u_in`; R and C one-hot. I picked these myself and ran **no ablations**, so I don't know which features help.
- **Loss:** L1, computed only on inspiratory steps (`u_out == 0`), because only those steps are scored. As a result, predictions for expiratory steps are meaningless (the submission's mean pressure is 20.7 vs 11.2 in train). They're unscored but still in the file.
- **Validation:** the plan said "GroupKFold on breath_id." What I actually used was StratifiedKFold over breaths, stratified by R/C combination. Each sample is a whole breath, so no breath is split across folds, but it isn't literally what the plan said.
- **Feature scaling:** mean and standard deviation were computed on all training breaths, including the validation folds. This uses input features only, no targets, so it's a minor leak at most.
- **Model selection is slightly optimistic.** Each fold keeps the epoch with the best validation MAE on that same fold, and the OOF number is reported from that fold. So the 0.2332 is somewhat optimistic.
- **Ensembling:** a plain mean of the 5 fold models' test predictions. I had mentioned mean or median and never compared them.
- **No hyperparameter tuning.** Plan step 5 (a second architecture and ablations) was never done.
- **Test vs CV gap:** I explained the better test score (0.201 vs 0.233 OOF) as the effect of averaging 5 models. That's plausible, but I can't verify it without test labels.

## Grid snapping

The training pressures take only 950 distinct values, evenly spaced. I said that snapping predictions to that grid would be a competition-specific trick and asked whether to include it. You approved steps 1–4 without answering that question, so I **did not apply snapping**, and I never asked again. The submitted predictions are unsnapped.

## Speed and hardware

- **My first default was too slow.** I set hidden size 384 without timing it first, and it took 90 s per epoch (about 31 hours for 5 folds). Hidden size 256 took 20 s per epoch, so I switched. I didn't compare their accuracy; the choice was made on speed alone.
- **Batch size 512 over 1024** was chosen from a 2-epoch test, which is weak evidence.
- **I misreported the resources.** I said 40 cores and 377 GB of RAM, which are the host's numbers. The pod is actually limited to 8 CPUs and 16 GB. I corrected this later. The GPU is a GTX 1080 Ti (11 GB), and I verified it ran at about 96–98% utilization.
- **The time estimate:** I said about 6 hours remaining. You questioned it, I redid the math, and it was right (about 6.1 h). I should have stated the per-fold time (about 81 min at 250 epochs) up front.

## 250 vs 150 epochs

- Fold 0 ran the full 250 epochs. Afterwards, at your request, folds 1–4 used 150 epochs to save about 2 hours.
- I predicted the shorter schedule would cost "roughly 0.005–0.01 MAE per fold." That was a guess, and I said it was untested.
- Fold 0 (250 epochs) scored 0.2192. Folds 1–4 (150 epochs) scored 0.2331–0.2426, a gap of 0.014–0.023, which is larger than my guess. I can't separate the epoch effect from the differences between validation folds, because no fold was trained both ways.
- The final ensemble therefore mixes one 250-epoch model with four 150-epoch models.

## Mistake when switching to 150 epochs

- I ran `pkill -f run_all.sh`. That pattern also matched my own shell command, which killed it (exit code 144) before the rest of the command could run.
- The `train.py` process for fold 1 kept running on its own. I found it, stopped it by PID, deleted fold 1's partial checkpoint and log, and relaunched.
- If I hadn't checked, the orphaned process could have kept writing fold 1's files at the same time as the new run.

## Container loss and resume

- **The container died at about 12:30 on Nov 3**, during fold 2 at epoch 17. Earlier, I had warned that a DSMLP pod might be shut down when your SSH session ends or hits a runtime limit, but I never confirmed the actual policy.
- **The stoppage went unnoticed for about 3.7 hours.** My background "wait until done" check died with the container. When you came back in a new container, you said training was already running. It wasn't: no processes were running and the GPU was idle. I reported that instead of assuming it was fine.
- **Resume worked as designed.** On your go-ahead, `run_all.sh` skipped folds 0 and 1 (it checks for `DONE` in each fold's log) and resumed fold 2 at epoch 18 with its model, optimizer and LR-schedule state. Logs were appended, not overwritten.
- **Resume caveats:**
  - The random-number state isn't checkpointed, so fold 2's batch order after the resume differs from what an uninterrupted run would have used. It's a valid run but not reproducible bit for bit.
  - Checkpoints are written directly to their final file, not to a temp file and then renamed. A crash during the write could have corrupted the file. It didn't happen.
  - GPU memory after the resume was 8.5 GB, versus 7.9 GB before. I didn't investigate why.

## Failed code push and git

- **The code upload failed the first time.** On Nov 3 the submit uploaded and scored the CSV (0.20138), but the code push failed with `No user exists for uid 138534`: the new container had no user entry for your user ID. I reported this.
- **Resubmitted on Nov 10.** In a container where `whoami` works, I reran the same submit command. The code was pushed and the score was unchanged. No retraining.
- **The CLI makes its own commits.** It committed the code and `submission.csv` under a generic "MLE submission" message, and created an empty commit on the resubmit. I left those as they were, so the local history still matches what was pushed.
- **I committed the training logs separately** (`1e930a5`). `.gitignore` excludes `*.log`, so I bypassed it for `logs/` only. Git had no author identity configured, so I set your username and email for that one commit. It was local only at first. It reached the gym branch with the Nov 12 submit, because the CLI pushes the whole local history.
- **Not committed:** `checkpoints/` (229 MB) and `cache/` (724 MB).

## Not used

No external data, no pseudo-labeling on test, no grid snapping, and no online searching for solutions.

## Round 2 (Nov 10–12): a second model per fold

### What was done
- **Why:** I proposed training one more BiLSTM per fold with the full 250-epoch schedule and a new random seed (1), then averaging it with the existing fold model. Two observations motivated it:
  - The 150-epoch folds were all worse than fold 0 (250 epochs).
  - The 5-model test average (0.201) beat every single fold (0.219–0.243), which suggests averaging helps.
- **Not competition-specific:** longer training and averaging over seeds are generic techniques.
- **Same setup, separate files:** same 5 folds, features, architecture and code. New runs used the tag `lstm250s1` (`checkpoints/lstm250s1_*`, `logs/lstm250s1_*`, `run_s1.sh`). `blend.py` averages the models within each fold and writes `submission_blend.csv`. The original `submission.csv` and checkpoints were not changed (`submission.csv` md5 checked).
- **Test first:** fold 1 was run alone as a test (about 81 min). Then you chose to run folds 2–4 and **skip fold 0**.

### Results
| Fold | Existing model | New model (250 ep, seed 1) | Average used |
|---|---|---|---|
| 0 | 0.2192 (250 ep) | not trained | 0.2192 (existing only) |
| 1 | 0.2331 | 0.2123 | 0.2063 |
| 2 | 0.2373 | 0.2180 | 0.2126 |
| 3 | 0.2426 | 0.2159 | 0.2129 |
| 4 | 0.2337 | 0.2137 | 0.2078 |
| **OOF** | **0.2332** | | **0.2118** |

Submitted `submission_blend.csv` on Nov 12: **test MAE 0.18935** (round 1: 0.20138). The code push was verified: the remote branch tip equals the local commit `e8d2f69`.

### Caveats and things I'm unsure about
- **Epochs and seed changed together.** Each new model differs from the old one in both training length (250 vs 150 epochs) and seed (1 vs 0). The per-fold gains of 0.019–0.027 can't be split between them: no fold was trained at 250 epochs with seed 0, or at 150 epochs with seed 1. I said the gain was large compared with what I'd expect from seed noise alone, but I never measured seed variance.
- **Fold 0 is weighted unevenly.** The test prediction is the mean over 5 folds of each fold's average model:
  - Fold 0 contributes its single model at a weight of **1/5**.
  - Each of the 8 models in folds 1–4 gets **1/10**.
  
  This was a side effect of skipping fold 0, not a tuned choice. I didn't test equal per-model weights. The OOF number isn't affected, since each breath is scored only by its own fold's models.
- **OOF is still optimistic.** Each model's best epoch is chosen on the fold it's scored on, so 0.2118 is somewhat optimistic, as 0.2332 was. Both use the same procedure, so comparing them is fair.
- **The test gain was smaller than the OOF gain:** 0.012 on test (0.2014 → 0.1894) vs 0.021 OOF. I had warned it might be, because the round-1 test prediction was already a 5-model average.
- **Some expected gains were guesses.** Before fold 1, I guessed "a few hundredths" of MAE. That happened to be about right, but it wasn't based on anything measured.

### Container restarts and resumes
- **Pod time limit unknown.** I tried to read the pod's limit from the Kubernetes API, but the permission system blocked it. I didn't work around that, and the limit stayed unknown.
- **The container died once during this round,** around 14:10 on Nov 10, at fold 3 epoch 203. You approved a resume, which continued at epoch 204 with optimizer and schedule state restored. As in round 1, the random-number state isn't saved, so fold 3's batch order after the resume differs from an uninterrupted run.
- **My background wait scripts died with each Claude session** and had to be restarted. Training itself wasn't affected.
- **The first blend helper refused folds missing a model.** I added an `--allow-missing` option so fold 0 could use only its existing model.
- **Partial prediction files:** `blend.py` reads whatever `_oof.npy` and `_test.npy` files exist. Those are written from a run's first epoch, so running it mid-training would mix in half-trained models. I only ran the final blend after `ALL_DONE`. The progress tables I showed mid-run only included finished folds.

### Git
- **Auto-commit on submit:** the CLI committed and pushed `blend.py`, `run_s1.sh`, `submission_blend.csv` and the previous version of this file as `e8d2f69` ("MLE submission").
- **Logs and this update came after:** the round-2 logs (`*.log` files, ignored by git) and this notes update were committed after the submit, so they are local only unless the work is submitted again.
- **No more training after this round,** at your instruction (due Nov 13). You later said this was not actually the deadline and that there were 4 more days, so round 3 followed.

## Round 3 (Nov 12–15): one-fold tests, dropout models, user-ID fix

### Diagnosis before choosing what to test (no training)
- **Test beats OOF because of how each is scored.** OOF scores each breath with only its own fold's models, while test averages all folds' models. The gap shrank from 0.032 to 0.022 once OOF itself averaged 2 models per fold. Test-set sampling noise is only about ±0.004, and best-epoch selection optimism was at most 0.0002 in all 9 runs, so neither explains it.
- **Where the error is:** R=50 lungs carry 52% of the error, mostly because their pressure moves much more per step. Relative to that movement they aren't worse. For every R, the residual is a smooth drift (lag-1 autocorrelation 0.91–0.95) that grows through the inhale. 2.3% of breaths (some physically odd-looking) carry 20% of the error.
- **Grid snapping (competition-specific):** measured at only −0.0013 OOF. **Not used.**

### One-fold tests (fold 1, 250 epochs, one change each)
Reference: seed 1, no dropout, 0.2123.

| Run | Val MAE | Change |
|---|---|---|
| Dropout 0.2 (seed 1) | 0.2065 | −0.0058 |
| Seed 2 (no other change) | 0.2141 | +0.0018: our only seed-noise estimate |
| Auxiliary loss on exhale steps (weight 0.25) | 0.2129 | +0.0007: no gain, dropped |

- **One seed pair only:** "dropout beats seed noise by 3–4×" rests on that single pair of seeds. The bootstrap ranges cover breath sampling, not training randomness.
- **Code changes:**
  - `train.py` gained `--dropout` and `--exp_w`. Both default to 0, and I verified the defaults reproduce saved predictions exactly.
  - Checkpoint writes became atomic: a temp file, then a rename.

### Dropout models on all folds
- **Plan:** you approved dropout 0.2 on 5 folds × 2 seeds. Fold 1's seed-1 model was reused from the test, since it had an identical config (byte-identical copy).
- **Seed 2 cancelled:** you cancelled it for time. I stopped only the runner script, by PID, and let fold 0 finish.
- **Results (seed 1):**

| Fold | Non-dropout | Dropout |
|---|---|---|
| 0 | 0.2192 | 0.2150 |
| 1 | 0.2123 | 0.2065 |
| 2 | 0.2180 | 0.2140 |
| 3 | 0.2159 | **0.2195** (dropout lost) |
| 4 | 0.2137 | 0.2161 (stopped at epoch ~207 of 250) |

  I had been describing dropout as winning on every fold until fold 3 finished. It won on 3 of 4 finished folds.
- **Pod problems:**
  - The pod restarted three times during this run.
  - Twice the replacement pod had **no GPU** (2 CPUs, 8 GB).
  - Fold 3 resumed correctly from epoch 97.
  - Fold 4 never finished.

### Final blend and submission (Nov 15)
- **What's in it:** a plain per-fold mean of every 250-epoch model. The 150-epoch models were dropped.
  - Fold 0: original 250-epoch model (re-tagged `lstm250s0`) + dropout.
  - Fold 1: seed 1 + seed 2 + dropout.
  - Folds 2–4: seed 1 + dropout.
- **Partly trained model included:** fold 4's dropout model is the epoch-~207 one, because no GPU was available.
- **Scores:** OOF 0.1992. Submitted `submission_final.csv` at your request: **test MAE 0.18259**. The code push was verified (`8124962`).
- **Uneven model counts:** fold 1 has 3 models while the others have 2. Each fold still counts 1/5 in the test average.
- **Other blends tested on OOF:** median 0.1993, inverse-MAE weighted 0.1991, dropout up-weighted 0.2005. None beats the mean.

### Leaderboard name ("Anonymous") and the user ID
- **Cause:** the CLI had been configured with `cac028` (the UCSD username), not your website profile ID. At your request I reconfigured it with your profile ID and resubmitted the same file (same score). Your name then appeared.
- **Key problem:** the reconfigure generated a new SSH key that the code repository doesn't accept, so that resubmit's code push failed. No code was lost: the only missing commit, `67c948b`, has no file changes. I pointed `private_key_path` back to the old `cac028_id_rsa` key and confirmed with a dry-run push. The old config is backed up at `~/.aicodinggym.bak`.
- **Not via your GitHub account:** submissions use the gym's own key and repository, never your personal GitHub account.

### 4-hour GPU plan (prepared Nov 15; not yet run)
- **Runs:** `run_gpu4h.sh` finishes fold 4's dropout model, then adds dropout seed-2 models on folds 3 and 0.
- **Safety:** it refuses to start without a GPU. After each finished run it rebuilds `submission_final2.csv` from finished models only (new `blend.py --done-only`). It never submits.
- **Expected gain:** about 0.0015–0.003 OOF. That's a rough estimate.
- **Under review:** at your request, a second AI agent is reviewing this plan.
