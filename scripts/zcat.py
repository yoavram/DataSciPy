# -*- coding: utf-8 -*-
"""
Extract a gzipped file and print its contents.

Created on Tue Feb  9 12:30:45 2016

@author: yoav@yoavram.com
"""
import sys
import gzip

def zcat(filename):
    with gzip.open(filename, mode='rt') as f:
        return(f.read())

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(
            """Usage: python {0} <filename>
            where <filename> is name of a text file compressed with gzip."""
            .format(*sys.argv)
        )
        sys.exit(1)
    filename = sys.argv[1]
    print(zcat(filename))
    sys.exit(0)
