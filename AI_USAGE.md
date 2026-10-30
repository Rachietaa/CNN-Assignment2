# AI Usage

## 1. AI coding tool used

**Claude Code** (Anthropic), used inside VS Code as a pair programmer for the whole project.

As the assignment puts it, the AI was my pair programmer, not my research scientist. It wrote code I
asked for, debugged errors, explained SSL methods and **suggested** options. I decided which SSL methods to
investigate, which experiments to run and how to make the comparisons fair, interpreted the results, and
made every Git commit myself, step by step. Each decision and the reason behind it is recorded in
[DECISIONS.md](DECISIONS.md); every result and its interpretation is in [EXPERIMENTS.md](EXPERIMENTS.md).

---

## 2. Representative examples of how AI assisted me

**1. Implementing three SSL methods from scratch.** No SSL library and no pretrained weights are used. The
AI implemented, at my request, SimCLR (two-view augmentation, projection head, NT-Xent loss), DINO
(multi-crop, EMA teacher, centering/sharpening loss) and MAE (ViT encoder, random patch masking, pixel
decoder), plus joint CE + SimCLR training. Before every long run it tested the new code on the CPU (loss
at chance level for random inputs, correct masking ratio, correct patch order, encoder weights loading
into the classifier).

**2. Diagnosing why DINO collapsed.** My first DINO run sat at a loss of exactly log(1,024) from epoch 3.
The AI measured that a randomly initialized CNN maps all images to nearly the same features (mean cosine
similarity 0.89), so after DINO's centering the teacher's targets are almost uniform from the first step.
That explained the failure and pointed to the fix (BatchNorm in the head and a sharper teacher).

**3. Monitoring and analysis tools.** A kNN monitor that measures representation quality during SSL
pretraining without training on labels; a view-preview tool that shows the augmented views before a long
run; a results collector (one table of every run with seed mean ± std); an ensemble evaluator.

**4. Running long experiments unattended.** Queue scripts that continue when one run fails, a "collapse
guard" that stops DINO automatically if it is still collapsed at epoch 25, and keeping the Mac awake with
the lid closed during multi-hour runs.

**5. Suggesting controls, which I then decided on.** The AI suggested a 300-epoch supervised run (is the
SSL gain just "more training"?) and seed repeats of the main comparison. Both changed my conclusions (see
Section 3).

---

## 3. Incorrect, ineffective or questionable AI suggestions

**Questionable: reporting the SSL gain from one seed.** After the first SimCLR fine-tune the AI summarized
the result as "+4.6 points over the supervised baseline". With three seeds of each, the supervised mean is
79.2 ± 1.8 and SimCLR 82.1 ± 1.3: the real gain at 100% labels is **+2.9**, because the first supervised
seed happened to be the worst of the three. The single-seed number overstated the effect by 60%.

**Missing control: crediting the best recipe's gain to SSL.** The AI built the best recipe by adding label
smoothing and Mixup/CutMix to the SimCLR fine-tune, then wrote in the report draft that SSL gave "+4.4 on test"
over the supervised baseline, without giving that baseline the same regularizers. A second review of the work
caught the confound. The matched control (supervised + the same regularizers) reached 80.6 ± 0.4 on test,
equal to the SimCLR recipe (80.2 ± 0.6): with all labels the gain came from regularization, not from SSL.

**Incorrect: SimCLR's blur copied from the paper unchanged.** The first SimCLR augmentation used the paper's
blur strength (σ up to 2.0 pixels), which was set for 224px images. On our 128px images it blurs ~1.75×
more relative to the content, and the view preview showed several unrecognizable views.

**Ineffective: DINO with paper-style defaults.** The first DINO configuration collapsed to uniform outputs
(Section 2). The AI's first fix attempt, a learnable scale for the prototype layer, had no effect at all in
a test run; only BatchNorm in the head plus a sharper teacher stopped the collapse — and even then DINO
reached only 39% kNN accuracy, against 69% for SimCLR at the same epoch.

**Ineffective on its own: a lower encoder learning rate for fine-tuning.** It made fine-tuning much faster
(79% after 10 epochs) but not better (82.3%, then declining as the model memorized the labels). It only
helped together with label smoothing and Mixup/CutMix (84.2%).

**Smaller errors:** it first said my `~/.venvml` environment had PyTorch (it does not; Assignment 1 used
`assignment1/.venv`); the classifier script buffered its output, so the log looked frozen for 20 minutes
until I asked why I could not see anything; a claim that validation loss "bottoms out early" was written
before checking the history file; several time estimates were off by 10–30%.

---

## 4. How I verified or modified the AI-generated solutions

| Issue | How it was verified | Change made |
|---|---|---|
| Single-seed SSL gain | Ran 2 more seeds of both the supervised baseline and the SimCLR fine-tune | Report 82.1 ± 1.3 vs 79.2 ± 1.8 (+2.9); every comparison in EXPERIMENTS.md rewritten with seed means |
| Best recipe's +4.4 credited to SSL | Supervised baseline retrained with the same regularizers, 3 seeds, val + test | Report rewritten: with all labels SSL gives no test gain over a regularized supervised model; final model unchanged |
| "SSL gain = more training?" | 300-epoch supervised control | 79.8%, the same as the 100-epoch mean: training longer does not explain the gain |
| Blur too strong | Looked at the view preview **before** the 1-hour SimCLR run | σ scaled to the image size (0.1–1.15); both previews kept as evidence |
| DINO collapse | Watched the per-image and batch entropy of the teacher's targets every epoch; measured the cause on a random network | BatchNorm head + teacher temperature 0.02; automatic collapse guard |
| DINO second attempt still weak | Stopping rule fixed **before** seeing the result (stop if kNN < 55% at epoch 100) | Stopped at epoch 100 (kNN 39.2%) instead of spending 1.5 h more |
| New code correct before long runs | CPU tests: loss at chance for random inputs, 75% masking exactly, patch order, train/val split identical to Assignment 1 index by index | Fixed a results-collector crash on older summary files |
| Numbers in the write-up | Recomputed from the saved `history.json` / `summary.json` files | Corrected the "val loss bottoms out early" claim and the best-epoch ranges |
| Test set used once | All model selection on the validation split; the final model (D12b) was fixed before the test set was evaluated | — |

---

## 5. Decisions I made rather than accepting the AI's recommendation

### Writing fresh code instead of copying the Assignment 1 pipeline

The AI's first plan was to copy the Assignment 1 code (data loading, training loop, configs) into this
repository and extend it. I chose to start a fresh codebase and reuse only the **data** and the **same
train/validation split** from Assignment 1. SSL needs different components anyway (two augmented views per
image, a contrastive loss, a separate pretraining loop), and a fresh repository contains only code that is
actually used in this assignment.

### Adding DINO as a second SSL method

The AI recommended SimCLR only, because of the deadline. I chose to run DINO as well, to test whether a more
complicated method is better. It produced one of the clearest findings of the project: DINO collapsed with
its standard settings and stayed far below SimCLR even after the fix.

### Adding a ViT with MAE, and testing a 60% masking ratio

The AI had ruled out masked image modeling because it needs a Vision Transformer, which is weak on 1,920
images. I asked to try it anyway. It answered the backbone question we had skipped: MAE helped the ViT more
(+5.0) than SimCLR helped the ResNet (+2.9), but the ViT stayed far below even the supervised ResNet. The
60%-masking ablation was also my idea.

### Not using the Assignment 1 model for the final system

The AI suggested pretraining my Assignment 1 SceneCNN with SimCLR as a possible best model. I decided to
keep ResNet-18 + SimCLR and use only Assignment 1's *training tricks* (label smoothing, Mixup/CutMix) in the
best recipe, not the Assignment 1 model.

### Running every planned experiment

When the overnight queue grew to ~5 hours, the AI offered to drop the augmentation ablation. I kept all of
it. The mild-augmentation run showed that the whole SSL benefit disappears without strong augmentation —
one of the main results.
