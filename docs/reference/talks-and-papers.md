<!--
SPDX-License-Identifier: MIT
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
-->

# Talks and Papers

This page collects publications, conference talks, and video presentations related to Iris.

## Papers

### Iris: First-Class Multi-GPU Programming Experience in Triton

**Authors**: Muhammad Awad, Muhammad Osama, Brandon Potter

**Published**: November 2025 · [arXiv:2511.12500](https://arxiv.org/abs/2511.12500)

Introduces the Iris framework, its programming model, and the SHMEM-like Remote Memory Access (RMA) APIs built on top of Triton. The paper demonstrates performance on AMD MI300X GPUs for communication-intensive workloads.

```bibtex
@misc{Awad:2025:IFM,
  author        = {Muhammad Awad and Muhammad Osama and Brandon Potter},
  title         = {Iris: First-Class Multi-{GPU} Programming Experience in {Triton}},
  year          = {2025},
  archivePrefix = {arXiv},
  eprint        = {2511.12500},
  primaryClass  = {cs.DC},
  doi           = {10.48550/arXiv.2511.12500}
}
```

---

### Eliminating Multi-GPU Performance Taxes: A Systems Approach to Efficient Distributed LLMs

**Authors**: Octavian Alexandru Trifan, Karthik Sangaiah, Muhammad Awad, Muhammad Osama, Sumanth Gudaparthi, Alexandru Nicolau, Alexander Veidenbaum, Ganesh Dasika

**Published**: November 2025 · [arXiv:2511.02168](https://arxiv.org/abs/2511.02168)

Presents a systems approach to reducing communication overheads in distributed large language model inference, leveraging Iris for fine-grained compute and communication overlap.

```bibtex
@misc{Trifan:2025:EMT,
  author        = {Octavian Alexandru Trifan and Karthik Sangaiah and Muhammad Awad and Muhammad Osama and Sumanth Gudaparthi and Alexandru Nicolau and Alexander Veidenbaum and Ganesh Dasika},
  title         = {Eliminating Multi-{GPU} Performance Taxes: A Systems Approach to Efficient Distributed {LLMs}},
  year          = {2025},
  archivePrefix = {arXiv},
  eprint        = {2511.02168},
  primaryClass  = {cs.DC},
  doi           = {10.48550/arXiv.2511.02168}
}
```

---

### Software Release

```bibtex
@software{Awad:2025:IFM:Software,
  author        = {Muhammad Awad and Muhammad Osama and Brandon Potter},
  title         = {Iris: First-Class Multi-{GPU} Programming Experience in {Triton}},
  year          = 2025,
  month         = oct,
  doi           = {10.5281/zenodo.17382307},
  url           = {https://github.com/ROCm/iris}
}
```

## Talks

### GPU Mode – Iris: Multi-GPU Programming Made Easier

**Date**: September 12, 2025

Iris was presented at the [GPU Mode](https://www.youtube.com/@GPUMODE) community meetup, covering the motivation, design, and performance results of the framework.

- 🎬 [Watch the talk on YouTube](https://www.youtube.com/watch?v=i6Y2EelEC04)
- 📄 [Download the slides](https://github.com/ROCm/iris/blob/main/docs/slides/Awad-Osama-Potter%20-%20Iris%20Multi-GPU%20Programming%20Made%20Easier%20(GPU%20Mode).pdf)

---

### AMD Distributed Inference Kernel Contest – Iris Introduction (Chinese)

**Date**: September 16, 2025

An introduction to Iris presented in Chinese for participants of the AMD Distributed Inference Kernel Contest.

- 🎬 [Watch on YouTube](https://youtu.be/wW14w1QNrY8)

---

### Iris All-Scatter Taxonomy

**Date**: August 14, 2025

A deep dive into the taxonomy of All-Scatter communication patterns and how Iris models them. See the accompanying [Taxonomy documentation](../conceptual/taxonomy.md) for written details.

- 🎬 [Watch on YouTube](https://youtu.be/fYMdPe9UpHE)
