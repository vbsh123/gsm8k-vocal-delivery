# Activation-inspired adaptive L31 steering

This is an exploratory attempt to improve the existing +4.3 pp method. The
existing mean vector, original results and model weights are unchanged.

`run_L31_adaptive.py` uses only the original extraction cohort (normally 425
pairs). At each neutral forward it computes a donor forward with angry audio
and **identical current answer embeddings**. It records the neutral L31 answer
mean and the angry-minus-neutral L31 answer-mean difference, without modifying
neutral generation. The first three questions are checked against fresh
unpatched neutral responses. Each question uses the same number of contributing
forwards as its saved matched-forward extraction cache. There is no 30-forward
cap and no collection of test-cohort angry activations.

Two things are fitted from those training pairs:

1. A fixed mean of the same-state differences, as a collection-method control.
2. A regularized predictor of the correction from the current neutral L31 answer
   mean. It uses a 32-dimensional PCA basis from training question means, then
   ridge regression with lambda 1. Every question receives equal total weight,
   regardless of its forward count. Neither answer correctness nor test labels
   enter fitting. A fixed 80/20 question split reports activation-prediction MSE;
   the final predictor is fitted on all extraction questions. Better activation
   prediction does not guarantee improved answer accuracy.

At inference only neutral audio is used. Before the final output normalization,
the predictor reads the current L31 answer mean and adds its predicted correction
to each answer token. Predictions therefore vary with the question and decoding
state. Feature values are clipped to five training standard deviations, and
correction norms are capped at twice the incumbent vector norm. Both the fixed
control and predictor use a common global scale so the training mean correction
matches the incumbent vector's norm. The predictor is tested at strengths 0.5
and 1, and the same-state fixed mean at strength 1.

All three interventions are evaluated on the existing disjoint 460+258 neutral
recordings, reusing compatible seed42 baselines and showing the saved original
L31 result alongside them. These evaluation cohorts are development data after
our method/layer exploration. Outcomes remain exploratory. The method may lose
accuracy: the predictor is trained on unsteered states and can encounter changed
states during steered generation. The half-strength run and correction cap are
predefined controls, not guarantees.

Run on the original steering machine:

```bash
git pull origin main
nohup .venv-diffa/bin/python -u run_L31_adaptive.py \
  > outputs/jenny-L31-adaptive.log 2>&1 < /dev/null &
tail -f outputs/jenny-L31-adaptive.log
```

It waits for existing GPU work and uses one model at a time. Collection releases
its model before evaluation loads a model. Per-question feature/target caches
resume, as do evaluation answers. With about 12,000 training forwards at hidden
size 4096, saved float32 features/targets need approximately 0.4 GiB. Model and
fitting checkpoint files are small. This is a substantial nightly run: collection
requires a donor forward for each contributing neutral forward, followed by up
to 2,154 new test generations. No waveform copies are created.

```bash
.venv-diffa/bin/python run_L31_adaptive.py --summary-only
```

Outputs are in `outputs/jenny-L31-adaptive/`: frozen training plan, `pairs/`
features/targets, `predictor.pt`, `fit-metrics.json`, per-cohort raw answers,
`combined-summary.json` and `status.json`.
