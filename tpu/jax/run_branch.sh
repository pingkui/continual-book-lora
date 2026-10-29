#!/bin/bash
# tau of the book subspaces under token-normalised subspaces (A4_gen70n, seed 0): book 2, then book 3 once per tau
# (here 0.97 and 0.95; 0.99, 0.995, 0.999 ran on 2026-10-24 in v10b) from the same post-book-2 state. Unpacked under /content/br.
set -u
cd /content/br/code
export SRC=/content/br/data JAX_COMPILATION_CACHE_DIR=/content/jaxcache PYTHONUNBUFFERED=1
RUNS=A4_gen70n BOOKS=2,3 BRANCH=1 BRANCHES=t97,t95 OUT=/content/results_branch.json python run_cl.py && echo "BRANCH DONE"
