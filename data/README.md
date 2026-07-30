# Data

This project uses the Oxford Flowers 102 dataset, loaded programmatically via
`tensorflow-datasets` (`tfds.load("oxford_flowers102")`). Raw and processed data are not
committed to version control (see `.gitignore`) to keep the repository lightweight.

- `raw/` — original downloaded/extracted dataset files (if cached locally instead of via tfds)
- `processed/` — intermediate artifacts (e.g., resized images, TFRecord shards, label maps)

To reproduce: running the data-loading notebook/script will populate these folders
automatically on first run.
