# Profile functions

Run profiler with
```
python -m cProfile -o filename.prof filename.py
```

See profiling results with (but first install with `python -m pip install snakeviz`):
```
snakeviz filename.prof
```

# Profile lines

Install line profiles with `python -m pip install line_profiler`.

Add `@profile` decorator to slow function.

Run
```
kernprof -l -v filename.py
```

# Profile memory

First install with `python -m pip install memory-profiler`.

Add a `@profile` decorator to a function.

Run 
```
python -m memory_profiler filename.py
```