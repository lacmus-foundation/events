import time
import numpy as np
from compute_overlap import compute_overlap


def compute_overlap_np(a: np.array, b: np.array) -> np.array:
    """
    Args
        a: (N, 4) ndarray of float [xmin, ymin, xmax, ymax]
        b: (K, 4) ndarray of float [xmin, ymin, xmax, ymax]

    Returns
        overlaps: (N, K) ndarray of overlap between boxes a and boxes b
    """
    N, K = len(a), len(b)
    overlaps = np.zeros(shape=(N, K))
    for n in range(N):
        a_area = (a[n, 2] - a[n, 0]) * (a[n, 3] - a[n, 1])
        for k in range(K):
            dx = min(a[n, 2], b[k, 2]) - max(a[n, 0], b[k, 0])
            if dx >= 0:
                dy = min(a[n, 3], b[k, 3]) - max(a[n, 1], b[k, 1])
                if dy >= 0:
                    b_area = (b[k, 2] - b[k, 0]) * (b[k, 3] - b[k, 1])
                    intersection = max(dx, 0) * max(dy, 0)
                    union = a_area + b_area - intersection
                    overlaps[n, k] = intersection / union

    return overlaps


def test_overlap_1():
    a = np.array([[1, 1, 3, 3]], dtype=np.float)
    b = np.array([[2, 2, 4, 4]], dtype=np.float)
    assert compute_overlap_np(a, b)[0][0] == 1. / 7


def test_overlap_0():
    a = np.array([[1, 1, 3, 3]], dtype=np.float)
    b = np.array([[3, 3, 4, 4]], dtype=np.float)
    assert compute_overlap_np(a, b)[0][0] == 0.


def test_overlap_n(a_len, b_len, box_size=100):
    a = np.random.randint(0, 3000, (a_len, 4))
    b = np.random.randint(0, 4000, (b_len, 4))
    a = a.astype(np.float)
    b = b.astype(np.float)
    a[:, 2] = a[:, 0] + box_size
    b[:, 2] = b[:, 0] + box_size
    a[:, 3] = a[:, 1] + box_size
    b[:, 3] = b[:, 1] + box_size

    t1 = time.time()
    o_np = compute_overlap_np(a, b)
    t2 = time.time()
    o_c = compute_overlap(a, b)
    t3 = time.time()
    assert np.array_equal(o_np, o_c)

    print('Numpy time = ', t2 - t1)
    print('C_ext time = ', t3 - t2)


if __name__ == '__main__':
    test_overlap_1()
    test_overlap_0()
    test_overlap_n(100, 5, 300)
