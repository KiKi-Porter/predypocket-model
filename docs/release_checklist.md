# Release checklist

Before publishing this repository:

- Replace remaining manuscript and author placeholders in `CITATION.cff`.
- Confirm that the bundled model weights and their checksums are present.
- Confirm that no private MD trajectories, labels, or cached RCSB downloads are committed.
- Run unit tests and source compilation from a clean environment.
- Run the toy smoke path: 3:3:4 sampling, local-contact data preparation,
  one-epoch MD model training, and MD inference.
- Confirm `README.md` uses the intended manuscript figure asset and that the
  figure-use policy is acceptable for the target GitHub release.
- Add final manuscript citation metadata after acceptance or preprint posting.
- Tag the GitHub release and attach any intended weight checksums.

Recommended weight checksums:

```bash
sha256sum weights/predypocket_model.*
```
