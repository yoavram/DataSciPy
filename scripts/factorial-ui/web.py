#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Licensed under the MIT license:
# http://www.opensource.org/licenses/MIT-license
# Copyright (c) 2016, Yoav Ram <yoav@yoavram.com>
import os
import json
from flask import Flask, jsonify


def factorial(n):
	'''Calculate the factorial of n using recursion.
	'''
	if n < 1:
		return 1
	else:
		return n * factorial(n-1)


app = Flask(__name__)


@app.route('/')
def root():
	return app.send_static_file('index.html')


@app.route('/<int:n>')
def calc(n=10):
	result = factorial(n)
	return jsonify(result=result)


if __name__=='__main__':
	app.run(host='127.0.0.1', port=5000, debug=True)
	app.logger.info("Server shuting down")
