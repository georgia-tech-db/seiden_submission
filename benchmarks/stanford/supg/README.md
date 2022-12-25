# Approximate Selection with Guarantees using Proxies

This is the official project page for "Approximate Selection with Guarantees using Proxies."

Please read our [paper](https://arxiv.org/abs/2004.00827) for full technical details.


# Requirements

You will need the following installed:
- python 3.x
- pandas
- numpy
- feather-format

You will also need to install the `supg` package. You can install by doing `pip install supg`, but we recommend using the latest version of the repository by running the following command in the root directory:
```
pip install -e .
```


# Reproducing experiments

In `supg/experiments`, run `experiments.py`. This will reproduce many of the end-to-end numbers in the paper.

To run your own experiments, create a CSV with the following columns: `id` (0-indexed and sequential), `label` (True or False), and `proxy_score` (float between 0 and 1). You can run the precision or recall target experiments as follows:
```
from supg import run_rt, run_pt

run_pt(csv_fname, budget, rt)  # Precision target
run_rt(csv_fname, budget, rt)  # Recall target
```
