# Timeline of the protocol

The order of these events is what makes the evaluation pre-registered: the test set, the baseline and the rule for choosing the adapter were all fixed before any test answer was judged. Times are commit timestamps from the development repository; this public repository was created from its final state.

| When | Commit | Event |
|---|---|---|
| 18 Oct 2026, 13:47 UTC | `9348df1` | Evaluation set frozen: 196 verified items, SHA-256 fingerprints recorded |
| 18 Oct 2026, 14:19 UTC | `66788b9` | Arm B's system prompt frozen after two dev-set tuning rounds |
| 18 Oct 2026, 14:29 UTC | `b0479ba` | Arm C pre-registered as the 500-example adapter |
| 18 Oct 2026, 14:49 UTC | `aae7124` | Training data finalised (generated after the freeze, checked against every evaluation item) |
| 18 Oct 2026, 15:13 UTC | `88e5cf5` | Deviation D-1 recorded: 6-epoch run, checkpoint chosen by dev joint success |
| 18 Oct 2026, 15:46 UTC | `60459ce` | D-1 rule applied: epoch 5 selected, before any test answer from the chosen adapter existed |
| 18 Oct 2026, 16:00 UTC | `c6138eb` | First test-set results committed |
| 18 Oct 2026, 17:25 UTC | `4bb6682` | Blind human labels committed (deviation D-2 records the safety criterion used) |
