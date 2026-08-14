# KRSL AI agent instructions

## Objective

Build a local-first KRSL recognizer, starting with isolated signs and a
signer-independent KRSL20 benchmark. Do not describe the first milestone as a
continuous sign-language translator.

## Non-negotiable rules

- Keep raw data immutable and out of Git.
- Split train, validation, and test data strictly by signer.
- Version every feature schema and record the config, seed, Git SHA, and
  metrics for each experiment.
- Keep data, features, models, training, inference, and UI/API separate.
- Report failures with sample IDs and reasons; never silently discard them.
- Return `UNKNOWN`/abstain at low confidence; do not force a label.
- Treat `kurshakuz/krsl-recogniton` only as a reference. Do not import or
  vendor its legacy MediaPipe/Bazel stack.

## Phase gates

Before moving to the next phase: run format, lint, tests, and a relevant smoke
test; save resulting artifacts; then update `STATUS.md` with commands,
results, known issues, and the next step.
