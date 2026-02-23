<!--
SPDX-License-Identifier: MIT
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
-->

# Talks and Papers

This page collects publications, conference talks, and videos related to Iris.

## Papers

### Iris: First-Class Multi-GPU Programming Experience in Triton

> Muhammad Awad, Muhammad Osama, Brandon Potter — *arXiv, November 2025*

Introduces the Iris framework and its SHMEM-like Remote Memory Access (RMA) APIs for multi-GPU programming inside Triton kernels, demonstrating programmability and competitive performance on AMD MI300X GPUs.

- 📄 [arXiv:2511.12500](https://arxiv.org/abs/2511.12500)
- 🔖 DOI: [10.48550/arXiv.2511.12500](https://doi.org/10.48550/arXiv.2511.12500)

**BibTeX**

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

> Octavian Alexandru Trifan, Karthik Sangaiah, Muhammad Awad, Muhammad Osama, Sumanth Gudaparthi, Alexandru Nicolau, Alexander Veidenbaum, Ganesh Dasika — *arXiv, November 2025*

Presents a systems-level approach for reducing communication overhead in distributed large language model inference, leveraging Iris for fine-grained GPU-to-GPU data movement.

- 📄 [arXiv:2511.02168](https://arxiv.org/abs/2511.02168)
- 🔖 DOI: [10.48550/arXiv.2511.02168](https://doi.org/10.48550/arXiv.2511.02168)

**BibTeX**

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

## Software Citation

If you use the Iris software directly, please also cite the software release:

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

---

## Talks and Videos

### Iris at GPU Mode — September 2025

Iris was presented at the GPU Mode meetup, covering the design of the RMA API, the symmetric heap, and performance results on multi-GPU workloads.

- 🎬 [Watch on YouTube](https://www.youtube.com/watch?v=i6Y2EelEC04)
- 📊 [Slides (PDF)](https://github.com/ROCm/iris/blob/main/docs/slides/Awad-Osama-Potter%20-%20Iris%20Multi-GPU%20Programming%20Made%20Easier%20(GPU%20Mode).pdf)

---

### Iris All-Scatter Taxonomy — August 2025

A deep-dive video on the taxonomy of multi-GPU programming patterns, with a focus on All-Scatter and GEMM + communication overlap.

- 🎬 [Watch on YouTube](https://youtu.be/fYMdPe9UpHE)
- 📖 [Taxonomy Documentation](../conceptual/taxonomy.md)

---

### Iris Presented in Chinese — September 2025

Iris was presented in Chinese for participants of the AMD Distributed Inference Kernel Contest.

- 🎬 [Watch on YouTube](https://youtu.be/wW14w1QNrY8)
