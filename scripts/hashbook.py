import hashlib
import binascii
import concurrent.futures
import time

names = [
    'Gulliver',
    'Alice in Wonderland',
    'Pride and prejudice',
    'Yellow wallpaper',
    'Metamorphosis ',
    'A Tale of Two Cities',
    'The Importance of Being Earnest',
    'Frankenstein'
]
filenames = {name: '../data/{}.txt'.format(name) for name in names}

def read_book(item):
    name, filename = item    
    with open(filename) as f:
        data = f.read()        
        return name, data

def hash_book(item, k=1024):
    name, data = item    
    # very slow function
    fingerprint = hashlib.pbkdf2_hmac('sha512', data.encode('utf8'), b'salt', 1000000)
    return name, binascii.hexlify(fingerprint).decode()

if __name__ == '__main__':
	# read books to memory
	with concurrent.futures.ThreadPoolExecutor(len(filenames)) as executor: # much faster than a single-threaded program
	    books = dict(executor.map(read_book, filenames.items()))   

	print("Single-thread")
	tic = time.time()
	results = map(hash_book, books.items())
	for name, fingerprint in results:
	    pass #print('Fingerprint for {} is "{}"'.format(name, fingerprint))	    
	toc = time.time()
	print("Elapsed time: {:.2f} seconds".format(toc - tic))

	print("Multi-thread")
	tic = time.time()
	with concurrent.futures.ThreadPoolExecutor(len(books)) as executor:
	    results = executor.map(hash_book, books.items())
	    for name, fingerprint in results:
	        pass #print('Fingerprint for {} is "{}"'.format(name, fingerprint))	    
	toc = time.time()
	print("Elapsed time: {:.2f} seconds".format(toc - tic))

	print("Multi-process")
	tic = time.time()
	with concurrent.futures.ProcessPoolExecutor() as executor:
	    results = executor.map(hash_book, books.items())
	    for name, fingerprint in results:
	        pass #print('Fingerprint for {} is "{}"'.format(name, fingerprint))	    
	toc = time.time()
	print("Elapsed time: {:.2f} seconds".format(toc - tic))