The exact plots in the paper are reproduced with a seed of 0, i.e. by running

```
python figures.py --seed=0
```

Try running things multiple times; with no explicit seed, numpy's generator is
seeded from /dev/urandom.

# Dependencies #

* numpy (tested with version 1.7.1, can be installed with pip)
* scipy (0.12.0, can be installed with pip)
* matplotlib (1.2.0, can be installed with pip)
* [Slycot](https://github.com/avventi/Slycot)
* [pydare](https://code.google.com/p/pydare/)

