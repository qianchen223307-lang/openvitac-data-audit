# Interface references and original contribution

Reference inspected: FVL-Repo/OpenViTac, commit `a3f4280ebca0f5c26cce67ac672cc6802a875f67`.

- [ACT process_data.py](https://github.com/FVL-Repo/OpenViTac/blob/a3f4280ebca0f5c26cce67ac672cc6802a875f67/policy/ACT/process_data.py): processed field paths, first eight joint coordinates, image naming and next-raw-step action pairing.
- [ACT utils.py](https://github.com/FVL-Repo/OpenViTac/blob/a3f4280ebca0f5c26cce67ac672cc6802a875f67/policy/ACT/utils.py): configurable camera/tactile inputs, per-stream resize at training read time.
- [envs/utils/data.py](https://github.com/FVL-Repo/OpenViTac/blob/a3f4280ebca0f5c26cce67ac672cc6802a875f67/envs/utils/data.py): raw field-stacked layout; rgb keys are JPEG streams rather than processed images.
- [Project page](https://fvl-repo.github.io/OpenViTac/) and [paper](https://arxiv.org/abs/2610.10384v1): research context only.

All checker, report, fixture and test implementations here were written independently. No upstream source, weights, sensor images, paper figures or performance numbers are redistributed. Repository ownership does not establish research authorship or affiliation. MIT in this repository applies only to the original code and generated assets here; it does not relicense upstream materials.
