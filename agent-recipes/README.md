# GPU4PySCF agent recipes

When writing or reviewing GPU code, use the
[coding style template](skills/gpu4pyscf-coding-style.template.md) for implementation
conventions. Its [Portability section](skills/gpu4pyscf-coding-style.template.md#portability)
covers CUDA, HIP, and SYCL kernels, Python backend dispatch, and build configuration.

To use these recipes regularly, append the following to an `AGENTS.md` in the root of
your checkout. Keep any instructions already there.

```markdown
For changes to GPU kernels, Python backend dispatch, or the GPU build,
read agent-recipes/skills/gpu4pyscf-coding-style.template.md.
```

Add your compiler paths, available devices, or preferred test commands below
those instructions. Referencing the shared template lets you pick up future updates.
If you prefer to edit the guidance itself, copy the template to a personal
file and point your local instructions there. To keep an untracked root
`AGENTS.md` out of Git status, add `/AGENTS.md` to `.git/info/exclude`.
