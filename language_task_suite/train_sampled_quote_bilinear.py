"""Train the shared paired-edit bilinear LM on sampled quote binding.

The cache supplies task-specific scope; the optimizer and behavioral metrics
are identical to the source-disjoint surname-copy control.
"""

from train_fineweb_surname_copy_bilinear import main


if __name__ == '__main__':
    main()
