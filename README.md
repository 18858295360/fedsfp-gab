# FedSFP-GAB simulation code

Simulation code and reproduction inputs for *Controlled Execution-Style Heterogeneity in Federated Imitation Learning*. The study pairs federated Streaming Flow Policy (FedSFP) with Global-Anchor Broadening (GAB) and compares policies across fixed client assignments.

| Benchmark | Start here |
| --- | --- |
| RoboMimic Square, Can, and Lift | [Training, evaluation, and split instructions](robomimic/README.md) |
| Push-T | [Training, evaluation, and archived Table 2 readout](pusht/README.md) |
| LIBERO-10 | [Training, evaluation, and split instructions](libero/README.md) |
| Partition diagnostics | [Trajectory separation and matched-action diagnostics](diagnostics/README.md) |

Each benchmark has its own Python environment and input layout. The split files describe membership by demonstration ID. Obtain the benchmark data from its upstream project and follow the corresponding preprocessing and ordering instructions before using those IDs. Evaluation requires checkpoints produced or supplied separately.

The Push-T trainer includes a fix for handing refreshed anchor caches to data-loader workers. Table 2 is recoverable from archived evaluation records. The saved Push-T GAB configuration records a refresh interval of five rounds, whereas the manuscript recipe gives one. The original training process's loaded code and arguments could not be established for each checkpoint, so the archived scores do not establish how the corrected implementation would perform.

This release covers the simulation components described above. The fixed-checkpoint mechanism probes, complete figure-generation pipeline, and physical robot protocol are outside its scope. Refer to each benchmark's README for its dependency and verification limits.

The Push-T setup assembles the verified runtime from a pinned public Streaming Flow Policy checkout and the study changes in this repository. Its archived Table 2 readout can be run directly. The LIBERO-10 setup uses the bundled original task-language embedding file.

The [original Streaming Flow Policy project](https://github.com/siddancha/streaming-flow-policy), [RoboMimic](https://github.com/ARISE-Initiative/robomimic), and [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) are external dependencies where specified. Their source, assets, and terms come from their respective upstream projects. Any redistributed third-party code retains its applicable notices alongside the corresponding benchmark. No new repository-wide license is declared here.
