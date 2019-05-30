""" Readable-einsum.

einsum 은 전체 iteration 을 돌려서 element multiplication 을 하여
해당하는 right matrix 에 박아넣는 것.

Reference: https://medium.com/datadriveninvestor/how-einstein-summation-works-and-its-applications-in-deep-learning-1649f925aaae
"""
import numpy as np


def get_el(term, value, order, pos):
    """ get element from value using current total position

    # example)
    # term: nr
    # value: [4, 7] matrix
    # order: nkrb
    # pos: [2, 3, 1, 4]
    # => term_pos = [2, 1] (from term nr)
    # => return value[2, 1]
    """
    # extract term-pos from pos
    term_pos = tuple(pos[order.index(ch)] for ch in term)
    return value[term_pos]


def einsum(s, *lvalues):
    """
    term: nr, rkb, nb 같은 각 matrix term
    ch: 각 matrix 에서 각 차원을 나타내는 n, r 등.
    value: 각 matrix 의 실제 값
    """
    if '->' not in s:
        # No arrow case: transpose?
        raise NotImplementedError()

    ### preprocessing ###
    left, right = s.split('->')
    lefts = left.split(',')

    # get dims
    dims = {}
    for term, value in zip(lefts, lvalues):
        for ch, dim in zip(term, value.shape):
            if ch in dims:
                assert dims[ch] == dim
                continue
            dims[ch] = dim

    ### compute einsum ###
    def compute(order, depth, pos):
        """ nested for iteration and compute einsum by recursion """
        if len(order) == depth:
            total = np.prod(
                [get_el(term, value, order, pos) for term, value in zip(lefts, lvalues)]
            )
            return total

        cur_ch = order[depth]
        cur_dim = dims[cur_ch]
        total = 0.
        for i in range(cur_dim):
            total += compute(order, depth+1, pos + [i])

        if depth == len(right):
            res[tuple(pos)] = total

        return total

    # right should be first in order
    left_ch_set = set(''.join(lefts)) # left ch set
    left_only = left_ch_set - set(right) # left only ch
    order = right + ''.join(left_only)

    # run
    res = np.zeros([dims[ch] for ch in right])
    total = compute(order, 0, []) # total = sum(res)

    return res


if __name__ == "__main__":
    N, R, B, K = 4, 3, 5, 7
    s = np.random.rand(N, R)
    w = np.random.rand(R, K, B)
    c = np.random.rand(N, B)
    print(einsum('nr,rkb,nb->nk', s, w, c))
    print(einsum('nr,rkb,nb->', s, w, c))
    print(np.einsum('nr,rkb,nb->nk', s, w, c))
    print(np.einsum('nr,rkb,nb->', s, w, c))
