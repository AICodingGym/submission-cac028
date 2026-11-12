# AI assistant decisions, mistakes and uncertainties

Ventilator pressure prediction, Nov 3–10, 2026. Written by the AI assistant (Claude) at the user's request.

**Final result:** test MAE 0.20138, out-of-fold CV MAE 0.2332, median baseline CV MAE 6.07.

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
- **I committed the training logs separately** (`1e930a5`). `.gitignore` excludes `*.log`, so I bypassed it for `logs/` only. Git had no author identity configured, so I set your username and email for that one commit. That commit is local only; it isn't on the gym branch.
- **Not committed:** `checkpoints/` (229 MB) and `cache/` (724 MB).

## Not used

No external data, no pseudo-labeling on test, no grid snapping, and no online searching for solutions.
