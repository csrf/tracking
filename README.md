# Experiments in tracking vehicles via keypoint clustering

**This is an experimental repository and does not reflect a proposed algorithm.**

This repository holds experiments to do with clustering detected DT-CWT
keypoints and descriptors into a) individula feature tracks and b) into
underlying vehicle motion.

## Installation and dependencies

The tracking code may be installed in the usual manner for Python projects:

```console
$ python setup.py install
```

The tracker expects keypoints in an HDF5 file as output from the
``displayVideoDTCWT`` program in [cldtcwt](https://github.com/csrf/cldtcwt/).

## Maintainers

* [Rich Wareham](https://github.com/rjw57/)
