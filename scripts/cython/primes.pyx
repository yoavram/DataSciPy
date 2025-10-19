def cfind_k_primes(int k):
    cdef int n_primes, candidate, p
    cdef int[1000] primes
    n_primes = 0  
    candidate = 2
    while n_primes < min(k, 1000):
        for p in primes[:n_primes]:
            if candidate % p == 0:
                break 
        else:
            primes[n_primes] = candidate
            n_primes += 1
        candidate += 1

    return [p for p in primes[:n_primes]] # convert primes from a cython type to a python list
