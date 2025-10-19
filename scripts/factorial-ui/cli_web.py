#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Licensed under the MIT license:
# http://www.opensource.org/licenses/MIT-license
# Copyright (c) 2016, Yoav Ram <yoav@yoavram.com>
import click
import requests

def factorial(n):
	r = requests.get('http://127.0.0.1:5000/{:d}'.format(n))
	if not r.ok:
		click.secho("Error: {}".format(r.reason), fg='red') # colors require to install colorama
		raise click.Abort()
	else:
		return r.json()['result']


@click.argument('n', type=int)
@click.command()
def main(n):
	f = factorial(n)
	print("n! =", f)
	

if __name__ == '__main__':
	main()