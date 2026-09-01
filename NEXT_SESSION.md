# Next session: Phase 3 — `sessions/audio.ipynb`

Branch `DL2026-gpu`, pushed, tracks `origin/DL2026-gpu`. Phases 2 and 7 are done
and committed with outputs. **Phase 3 is the last item on the GPU handoff.**

## Read, in this order

1. `DL2026_GPU_HANDOFF.md` §4 — the spec, *and* the "Verified on the GPU box"
   subsection at the end of it. That subsection is the important part.
2. `DL2026_PLAN.md` §4 — the same spec from the plan's side; §4b is done, ignore it.
3. `CLAUDE.md` — house style.
4. `sessions/transfer.ipynb` — Phase 2's notebook. Phase 3 mirrors its structure
   on a different modality, so copy its shape: cached-embeddings probe that
   trains live, fine-tune behind a checkpoint, comparison table, discussion
   written around measured numbers.

## Start here

There is no long-pole download this time — ESC-50 is already on the box
(`data/ESC-50-master`, 2,000 wav files). Start by fixing the blocker:

```python
# sessions/audio.ipynb, cell 5. Returns int16; librosa 1.0 refuses it.
def load_wave(i):
    rate, wave = scipy.io.wavfile.read(audio_folder + metadata['filename'][i])
    return wave.astype(np.float32) / 32768.0, rate   # <-- verified fix
```

**The notebook does not currently run.** `librosa` 1.0.0 raises
`ParameterError: Audio data must be floating-point`, so nothing below cell 13 has
executed in this environment. Confirm the fix works end to end before you design
anything.

## The five defects (detail in `DL2026_GPU_HANDOFF.md` §4)

1. **Fatal** — int16 audio into librosa 1.0. Fix above.
2. `librosa.amplitude_to_db` applied to a *power* spectrogram; should be
   `power_to_db`. Every dB value is currently halved.
3. **The train/validation split leaks.** `validation_split=0.1` splits over
   ~1s overlapping *segments* cut from the same 5s clip, so windows of one
   recording land on both sides and the reported ~55% is optimistic. ESC-50 ships
   5 official folds of exactly 400 clips (`esc50.csv` has a `fold` column) —
   split on those. **Do this first: it moves the baseline the whole session is a
   comparison against.**
4. The history plot reads `history['acc']` / `history['val_acc']`; Keras 3 uses
   `accuracy` / `val_accuracy`, so cell 37 raises `KeyError` as committed.
5. Saves `.h5` (branch idiom is `.keras`), and says *test* where the branch swept
   to *validation*.

## One thing in the spec that is wrong

§4 says to build the three channels from different window sizes **and hop
lengths**. Measured: `hop_length` 512/1024/2048 gives 431/216/108 frames, so they
cannot be stacked. Varying `n_fft` (1024/2048/4096) at a fixed `hop_length` keeps
all three at 431 frames and stacks directly. Pick one, say which in the notebook.

## Verified for you

- Environment: keras 3.15.1, jax 0.11.1, backend `jax`, `gpu`, 2x RTX A4000
  16 GB, `.venv` at Python 3.12.13, librosa 1.0.0.
- `EfficientNetV2S(weights='imagenet', include_top=False, pooling='avg')` loads,
  gives 1280-d embeddings, and leaves the spatial axes free — spectrogram-shaped
  input is accepted directly.
- Mel-spectrogram prep is **7 ms/clip**, ~12 s for all 2,000 single-threaded.
  Cell 18's "this can take several minutes" is stale.
- The existing segmenter gives **17 segments/clip, 34,000 total**; as float32
  with the delta channel that is **1.54 GB** (keep it off float64, which is 3 GB).
- No ESC-50 checkpoint exists on the box — the baseline must be trained and saved.
- Disk 27 GB free (97% full). `data/CUB_200_2011/attributes/` (70 MB) is unused
  and can be deleted if you need room.
- `sessions/audio.ipynb` is 35 MB — embedded audio and figure outputs. Expect
  slow diffs; the blob is already in `master`'s history so it costs nothing new.

## Conventions

- **Anything stated as a mechanism in a discussion cell must be measured, or
  labelled unmeasured.** GPU runs here are minutes. See the unfreeze-depth table
  in `sessions/transfer.ipynb` and the model-comparison table in
  `sessions/flow.ipynb` for the pattern.
- **Do not rig the result.** §4 is explicit that the frozen probe may land
  *below* the from-scratch CNN, because ImageNet is a weaker prior for
  environmental audio than AudioSet would be. If that is what happens, write the
  discussion around it — "the domain gap is real and pretrained is not
  automatically better" is the more valuable lesson.
- Report a range over seeds, not a single number.
- `DL2026_PLAN.md` **is now editable from this branch** (its correction 18).
  Record Phase 3 in its §0a progress log the way Phases 2 and 7 are recorded, and
  add a numbered correction for anything in the handoff that turns out wrong.

## Deliverable

Notebook executes top to bottom from a clean kernel, committed **with outputs**,
within its 1 academic hour budget. Report measured ESC-50 accuracy for
from-scratch vs probe vs fine-tune, whether the probe beat from-scratch, the
fold split you used, and the from-scratch baseline after the leak is fixed
against the ~55% the leaky split reported. Full list in
`DL2026_GPU_HANDOFF.md` §6; acceptance checks in §7.

## Still open, beyond Phase 3

Day 4 is 2 academic hours short (`DL2026_PLAN.md` correction 10). Phases 7 and 8
did not close it — the flow session stays at 1.5 AH and the autoencoder leftovers
are worth about 0.5 AH. Needs a decision, not more notebook work.
