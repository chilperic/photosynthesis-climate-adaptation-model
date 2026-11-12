# FAIR4RS implementation and open requirements

| Principle | Implemented evidence | Still required |
| --- | --- | --- |
| Findable | CITATION.cff 1.2, CodeMeta 3, repository URL and version | Register an immutable version DOI after release and authorship review |
| Accessible | Public Git repository, source history, documented entry points | Deposit a durable release independent of GitHub |
| Interoperable | CSV/JSON results, explicit model/state names, RO-Crate 1.3 metadata | Verify every physical unit and domain annotation against primary sources |
| Reusable | Existing MIT code license, input checksums, tests, examples, migration notes | Confirm inherited data rights and regenerate published figures |

This is an implementation record, not FAIR certification. No DOI, ORCID,
authorship, dataset license or experimental validation has been invented.
Data checksums identify files; they do not establish correctness or ownership.
The citation author preserves the repository's existing author record; retain
all historical acknowledgements and review contributor roles before a release.

## Layout and provenance

The supported package is `src/` (Python) or `R/` (R). Tests live in `tests/`,
run configurations in `examples/`, and methods/migration notes in `docs/`.
Original notebooks, legacy equations and input data remain in their existing
locations so their links and provenance remain available. Generated runs go
to `runs/`; exported CSV/JSON plus a run manifest should accompany a paper.
Do not commit environments, credentials or generic caches. Do not blanket-ignore
PDF, SVG, PNG, HTML or scientific data formats: they may be research sources.

## References

- [FAIR4RS principles v1.0](https://doi.org/10.15497/RDA00068)
- [Citation File Format](https://github.com/citation-file-format/citation-file-format)
- [CodeMeta terms](https://codemeta.github.io/terms/)
- [RO-Crate 1.3](https://www.researchobject.org/ro-crate/specification/1.3/)
