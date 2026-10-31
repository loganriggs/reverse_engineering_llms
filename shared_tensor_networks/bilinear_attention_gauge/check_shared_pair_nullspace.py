"""Exact three-product construction verifies the joint-nullspace gauge."""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

from shared_pair_gauge import joint_nullspace_gauge


def check():
    truth=np.eye(3,8)
    common=np.eye(3)
    rare=common[[1,2,0]]
    matrix=np.zeros((6,8))
    matrix[:3,:3]=common
    matrix[3:,:3]=rare
    q=np.zeros((2,3,3))
    q[0,0,1]=q[0,1,0]=.5
    q[1,1,2]=q[1,2,1]=.5
    for seed in range(100):
        rng=np.random.default_rng(seed)
        base=Rotation.random(random_state=rng).as_matrix()@truth
        d,diagnostic=joint_nullspace_gauge(matrix,q,base)
        similarity=np.abs(d@truth.T)
        axis,family=linear_sum_assignment(-similarity)
        assert np.min(similarity[axis,family])>1-1e-12
        assert np.max(np.abs(d.T@d-truth.T@truth))<1e-12
        assert abs(diagnostic['null_vector_volume']-1)<1e-12
    print('Joint-nullspace gauge recovers the ideal three-product axes in 100 random gauges.')


if __name__=='__main__':check()
