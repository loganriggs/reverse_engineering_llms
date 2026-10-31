"""Evaluate the declared natural-output/edit-response mixture grid."""

from flagship_head_joint_moment_fit import LAMBDA_VALUES
from flagship_output_aware_head_eval import run


if __name__=='__main__':
    run(basis_filename='head_joint_moment_fit_bases.npz',
        methods=tuple(f'lambda_{value:g}' for value in LAMBDA_VALUES),
        output_filename='head_joint_moment_eval.json',check_old=False)
