#!/usr/bin/env python
# coding: utf-8
# https://blog.finxter.com/python-cprofile-a-helpful-guide-with-prime-example/

import random

def guess():
    ''' Returns a random number '''
    return random.randint(2, 1000)

def is_prime(n):
    ''' Checks whether n is prime '''
    for i in range(2, n):
        for j in range(2, n):
            if i * j == n:
                return False
    return True

def find_primes(num):
    primes = []
    while len(primes) < num:
        p = guess()
        if is_prime(p):
        	primes.append(p)
    return primes

print(find_primes(100))