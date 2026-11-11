# LIBERO-10 task embeddings

[`libero_10_lang.npz`](libero_10_lang.npz) is a byte-for-byte copy of the language input recorded by the archived LIBERO-10 checkpoints. Its SHA-256 is `c392dc42debb7ca1eb5fbf1d87870386d45f0e5f191fa162dce72e7f19c3e56f`.

The archive contains two NumPy arrays. `tasks` holds ten fixed-width Unicode task identifiers, with the exact keys used by the LIBERO-10 HDF5 files and simulator. `emb` holds the corresponding ten finite, 384-dimensional float32 vectors in the same row order. Load the archive with `numpy.load(path, allow_pickle=False)`.

The task identifiers come from *LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning* by Bo Liu et al. The benchmark's dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The embedding vectors are derived features of the task instructions. [LIBERO dataset attribution](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/README.md#license) · [Task identifiers](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/libero/libero/benchmark/libero_suite_task_map.py)

The archived checkpoint arguments and matching archive digest identify the asset used in the experiments. The encoder model, revision and generation settings remain unverified, so the preserved vectors are the reproducibility input.
