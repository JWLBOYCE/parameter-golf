# AGENTS

## RunPod Cost Control

- After every remote training run on RunPod, stop the pod as part of the same task once logs and needed artifacts have been collected.
- Prefer `runpodctl pod stop <pod-id>` immediately after validation, packaging, or log collection is complete.
- If a pod must stay up for a follow-up command, state that explicitly first; otherwise default to stopping it to minimize cost.
- Treat stopping idle pods as mandatory, not optional.

## Run Logging And Git Hygiene

- After every remote training run, append the result to the in-repo run ledger with the config, code changes, performance, and a comparison against the best public run at that time.
- Keep a human-readable top-3 summary of the best runs so far and prune non-top-3 local artifacts when it is safe to do so.
- After a completed run and ledger update, push the relevant code and run-log changes to GitHub, but never include secrets, API keys, or home-directory config files.
- Before pushing, check `git status` and confirm only intended repo files are staged.
