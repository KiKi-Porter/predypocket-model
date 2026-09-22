# Contributing

Contributions are welcome for reproducibility fixes, data-format support,
documentation, and model validation.

Before opening a pull request:

```bash
PYTHONPATH=src python -m unittest discover -s tests
python -m py_compile src/*.py
```

Please keep changes scoped. Avoid committing generated datasets, trajectory
files, or TensorFlow checkpoint binaries.
