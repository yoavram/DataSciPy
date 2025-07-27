# -*- coding: utf-8 -*-
"""
Extract a gzipped file and print its contents.

Created on Tue Feb  9 12:30:45 2016

@author: yoav@yoavram.com
"""
import gzip
import click

@click.command()
@click.argument('filename', type=click.Path(exists=True, readable=True))
def zcat(filename):
    with gzip.open(filename, mode='rt', encoding='utf8') as f:        
        print(f.read())

if __name__ == '__main__':
    zcat()
