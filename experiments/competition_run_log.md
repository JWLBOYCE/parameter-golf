# Competition Run Log

This file tracks local and remote competition runs, including what changed, the measured result, and how that result compared with the best public run known at the time.

## Top 3 Runs

| Rank | Run | Final BPB | Gap To Public Best | Bytes | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | 2026-03-20_mainline_smoke_1xh100_120s_1shard | 2.978911 | 1.833651 | 5,044,461 | End-to-end 1xH100 smoke succeeded, but the roundtrip gap was very large and the artifact used only about 5.0MB of the 16MB budget, so this run is useful as a systems checkpoint, not a competitive score. |

## All Runs

| Timestamp | Run | Variant | GPU | Train Time (s) | Shards | Pre BPB | Final BPB | Public Best | Gap | Bytes | Changes | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 2026-03-20T09:30:10Z | 2026-03-20_mainline_smoke_1xh100_120s_1shard | mainline | 1x H100 SXM | 120.0 | 1 | 2.492400 | 2.978911 | 1.145260 | 1.833651 | 5,044,461 | Ran the records mainline workbench on RunPod after fixing the TorchDynamo import alias in the records trainer; installed zstandard on the pod so zstd export could complete. | End-to-end 1xH100 smoke succeeded, but the roundtrip gap was very large and the artifact used only about 5.0MB of the 16MB budget, so this run is useful as a systems checkpoint, not a competitive score. |
