# Emotion-basis provenance

Byte-identical copies from ICMI-022 *through-the-valley* (`through-the-valley/vectors/`),
2026-08-21:

```
8fe7eb45d115e9c7cef9eeca1850390e502aa676029a9942e35959c417a1b37e  emotion_vectors_qwen27b_best.pt
89ec7f8f1ac86f31db2f1424a290b525f0e83e0f8b1efc5467b9adc711a45a3b  meta_qwen27b.json
```

- 171 emotion-direction vectors, Qwen 3.5 27B, decoder layer 53, hidden 5120,
  L2-normalized rows aligned to `meta_qwen27b.json`'s `emotion_order`.
- Extraction follows Lindsey et al. (2026): 6 first-person narratives per
  emotion, mean-pooled residual (skip first 4 tokens), PCA denoising against
  24 neutral prompts (50% variance), difference-of-means, L2 norm.
- Valid ONLY at model revision `fc05daec18b0a78c049392ed2e771dde82bdf654`
  (pinned in `src/config.py`).
- Full extraction details: through-the-valley `vectors/PROVENANCE.md` and
  paper §3.1–3.2.
