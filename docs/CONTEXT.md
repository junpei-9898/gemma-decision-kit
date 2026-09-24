# Text context limits and measured accuracy — v0.3.0

Text input is extended to **262143tokens per question**, within Gemma4's native262144total context, reserving one decision output token. Default input limit is65535;128K and256K are opt-in. This release extends mechanical input capacity; it does **not** fix long-context accuracy.1M is unsupported. [Pinned model configuration](https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4/blob/a19cfe00be84568a6867111c9a68c9c44fdcffe6/config.json).

```sh
# Add to the documented predict/serve command:
--max-input-tokens 131071  #128K total context
--max-input-tokens 262143  #256K total context
```

Restart to change capacity. [Complete installation instructions](INSTALL.md). With `--media`, the previous8192expanded-input limit and16384internal context remain unchanged. Text state is limited to2Mcharacters, JSON to8MiB UTF-8 bytes and all prepared questions together to524288input tokens. Question/criteria/chat framing count toward the limit. Inputs are rejected before any inference if a limit is exceeded; they are not truncated.

## Same nine cases at six input lengths

![Accuracy and latency by context length](assets/context-accuracy.png)

| Input | Actual prompt tokens | State characters | Correct / all | Agreement | Median HTTP seconds† |
|---|---:|---:|---:|---:|---:|
|short|118–141|35–74|7/9|77.8%|0.072|
|8k|8,151–8,191|11,211–11,270|8/9|88.9%|1.534|
|32k|32,731–32,767|45,548–45,596|7/9|77.8%|7.927|
|64k|65,499–65,522|91,258–91,297|7/9|77.8%|20.717|
|128k|131,047–131,070|182,706–182,745|7/9|77.8%|60.795|
|256k|262,104–262,141|365,733–365,792|7/9|77.8%|199.206|

†256K timing is from a separate concurrency-authorized resumed process and is reference-only. Earlier lengths were measured in isolation.

The short control scored7/9; near256K scored7/9, a **+0.0percentage-point** change on these fixed cases. Inspect intermediate points: a larger context does not necessarily cause a monotonic decrease. These rates are **not directly comparable to93.3% (112/120)** in the historical OSS comparison, which used different cases and prompts. They are also not evidence that the context-setting change itself damaged previous short-input behavior: the same231legacy decisions and probabilities matched exactly in both the64K and256K process configurations.

The nine questions, evidence and AI-provisional labels were frozen before predictions. Supported/refuted/insufficient each have three cases; evidence positions span beginning/middle/end. Only the amount of irrelevant background changes. The background uses varied but templated fictional business records, **not a representative natural-document corpus**. All54outputs are scored; no incorrect case was removed or relabeled. Each case was measured once per length; timing is the median across9different cases, **not9repeats of one case**. It includes incorrect answers and is not a time-to-correct-answer metric. The small sample and shared generator do not support population-level accuracy or statistical significance claims.

Label caveat: `draft_insufficient` describes an undecided formal start date, and a `refuted` reading may also be reasonable. Its precommitted `insufficient` label remains in the denominator pending independent human review. A single case changes this9case rate by11.1percentage points. The results are agreement with provisional labels, not audited human-gold accuracy.

## Previous stress test remains visible

The orange line preserves WI056's seven repeated-negative probes: short7/7 and8K/32K/64K4/7. Its filler repeatedly states that an individual record has no approval information, which may bias decisions toward `insufficient`; the cause is unproven. These are separate inputs, with65536internal context and3GiB KV.128K/256K stress points were not measured and are not extrapolated. [Original failed-gate record](CONTEXT_TRIAL_WI056.md). The user subsequently authorized mechanical extension while publishing quality limitations, rather than requiring every new probe to be correct.

## Runtime and memory

The blue curve uses a **fixed262144internal context and8GiB FP8 KV at every length**. Short through128K ran in the initial process;256K ran in a separate resumed process after a foreign-container interruption. The user explicitly allowed concurrency for the resumed stage. This keeps capacity fixed, but256K timing is reference-only and must not be treated as an isolated same-session speed comparison. NVIDIA GB10 / EdgeXpert, Linux ARM64,8CPU quota,4OMP threads, pinned NVFP4 revision `a19cfe00be84568a6867111c9a68c9c44fdcffe6`, vLLM `0.26.1.dev0+gf2654939e.d20260726`. Same frozen optimized kernels/prompt template; no fine-tuning, new quantization or RoPE extrapolation. Prefill chunks8192, sequential questions, prefix cache disabled.

Latency is same-container loopback HTTP including input validation, tokenization, inference and JSON. The model was loaded and a short warmup excluded from the curve; any length-specific first-use work remains included. Lengths ran in increasing order, not randomized; warmup and legacy parity calls remain in the raw budget. Every accepted input was checked against the backend's returned prompt token IDs. Largest actual prompt262141tokens; CPU tests cover configured262143boundary. Oversized requests returnedHTTP400 at each tier. The interrupted256K request has no scored output and remains in the conservative budget; the resumed nine precommitted cases supply the complete256K measurement.

| Text process | Input limit / question | KV allocation | Peak allocated | Peak reserved |
|---|---:|---:|---:|---:|
|65,536 total context|65,535|3GiB|22.005GiB|23.912GiB|
|131,072 total context|131,071|4GiB|23.009GiB|23.629GiB|
|262,144 total context|262,143|8GiB|27.005GiB|27.633GiB|

KV allocation is part of total allocated memory. Reserved includes allocated: **do not sum these columns**. The256K row covers the resumed256K cases and startup;64K/128K rows are separate process checks. These are peak PyTorch GPU figures, not total system consumption or certifications of minimum discrete VRAM. GB10 shares CPU/GPU memory. No new swapout or OOM was observed in the completed trials. Foreign workload presence was monitored; the initial process stopped on a foreign container, while the resumed stage explicitly permitted concurrency. See raw safety summaries for observed overlap. More room is needed for OS/runtime/loading than a weight-file size alone suggests.

The historical50k-character timing remains8.686s for35257tokens in a different native-runner fixture. That is separate from the HTTP curve above; neither figure implies all50k-character documents have the same accuracy or latency.

## Future accuracy work

- Independently audit labels and expand to a larger held-out set of natural Japanese documents, retaining this stress set and all failures.
- Separate evidence-position effects, irrelevant-record density, repetition and logical reasoning errors with paired controls.
- Compare the optimized path against the unmodified reference, and evaluate KV precision under matched inputs before attributing losses to quantization or kernels.
- Evaluate candidate improvements on the same frozen cases without truncating content, selectively dropping errors or optimizing only for these nine questions.

No long-context accuracy improvement is claimed here. Native capacity and task quality are separate properties.

## Evidence and reproduction

[JSON summary](context-accuracy-results.json), [all54case results CSV](context-accuracy-cases.csv), [frozen input/token hashes](context-fixture-manifest.json), [SVG](assets/context-accuracy.svg), [PNG](assets/context-accuracy.png). The [nine frozen cases](../benchmarks/context_cases.json) and [deterministic fixture generator](../benchmarks/context_fixture.py) are public. Use the pinned tokenizer and `build_body(tokenizer, case, target_tokens)` with targets0/8191/32767/65535/131071/262143, then submit each generated body to the documented loopback API. Compare prompt counts/hashes to the CSV. Graphs can be rebuilt using `python benchmarks/plot_context_curve.py` with matplotlib installed separately; it is not a runtime dependency. Private legacy inputs and model weights are not redistributed.

## Per-case outcome across lengths

| Case | Expected | Short | 8K | 32K | 64K | 128K | 256K |
|---|---|---|---|---|---|---|---|
|date_supported|supported|supported ✓|supported ✓|insufficient ✗|insufficient ✗|insufficient ✗|insufficient ✗|
|delivery_refuted|refuted|refuted ✓|refuted ✓|refuted ✓|refuted ✓|refuted ✓|refuted ✓|
|budget_insufficient|insufficient|insufficient ✓|insufficient ✓|insufficient ✓|insufficient ✓|insufficient ✓|insufficient ✓|
|rule_supported|supported|supported ✓|supported ✓|supported ✓|supported ✓|supported ✓|supported ✓|
|arithmetic_refuted|refuted|supported ✗|insufficient ✗|insufficient ✗|insufficient ✗|insufficient ✗|insufficient ✗|
|draft_insufficient|insufficient|refuted ✗|insufficient ✓|insufficient ✓|insufficient ✓|insufficient ✓|insufficient ✓|
|privacy_supported|supported|supported ✓|supported ✓|supported ✓|supported ✓|supported ✓|supported ✓|
|revision_refuted|refuted|refuted ✓|refuted ✓|refuted ✓|refuted ✓|refuted ✓|refuted ✓|
|approver_insufficient|insufficient|insufficient ✓|insufficient ✓|insufficient ✓|insufficient ✓|insufficient ✓|insufficient ✓|

Equal aggregate scores can hide different errors. These per-case results retain every precommitted label, including the ambiguous draft-date case.
