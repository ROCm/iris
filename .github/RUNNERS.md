# CI runner hardware

The `linux-mi325-8gpu-ossci-rad` label is historical. The pool behind it is
**8x AMD Instinct MI350X (gfx950, 256 CUs)** -- runner names are
`iris-mi350x-N`, and every job's `rocm-smi` output reports
`AMD Instinct MI350X (gfx950:sramecc+:xnack-)`.

Do not infer the GPU model or gfx arch from the label. Anything built for the
runner's GPU (device bitcode, `--offload-arch`, CU-count-dependent arguments
such as `--gemm_sms`) has to target gfx950.

The label itself is attached when the runner is registered, so renaming it
needs the runner/org admins; `runs-on` must keep matching the existing label or
jobs queue forever. When the label is renamed, update every `runs-on` that
references it and delete this file.

The Copilot agent runner (`[self-hosted, copilot, apptainer, iris]`) is a
separate pool -- check `rocm-smi` there rather than assuming.
