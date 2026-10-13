"""ck3dna: train and run a Crusader Kings 3 character DNA predictor from a
portrait photo.

Pipeline overview (see README.md for full details):

    alignment -> data_processing -> preprocess_tensors -> training -> infer

This project is a clean rebuild of
`Hakim1625/ck3-image-portrait-modeling <https://github.com/Hakim1625/ck3-image-portrait-modeling>`_,
with the gaps in the original filled in (a first-class inference CLI, the
missing image/dna tensor preprocessing step, the `gene_dicts.json` /
`default_dna.txt` dependency files, a regressor scaling-bug fix, and a port
to the current pytorch-lightning hook API). See the README for full credit
and a list of the specific fixes/assumptions made.
"""

__version__ = "0.1.0"
