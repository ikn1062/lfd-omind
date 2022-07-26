def recc(K, k_arr, n, f):
    if n > 0:
        for k in range(K):
            recc(K, n - 1, k_arr + [k], f)
    else:
        f(k_arr)
