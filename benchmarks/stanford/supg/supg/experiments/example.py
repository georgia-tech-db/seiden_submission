import supg.datasource as datasource
from supg.sampler import ImportanceSampler
from supg.selector import ApproxQuery
from supg.selector import RecallSelector
from supg.selector import ImportancePrecisionTwoStageSelector
from supg.experiments.trial_runner import TrialRunner


def run_helper(csv_fname, budget, target, qtype, delta=0.05, verbose=False):
    source = datasource.load_csv_source(csv_fname)
    sampler = ImportanceSampler()
    query = ApproxQuery(
            qtype=qtype,
            min_recall=target, min_precision=target, delta=0.05,
            budget=budget
    )

    if qtype == 'rt':
        selector = RecallSelector(query, source, sampler, sample_mode='sqrt', verbose=False)
    elif qtype == 'pt':
        selector = ImportancePrecisionTwoStageSelector(query, source, sampler)
    else:
        raise NotImplementedError

    trial_runner = TrialRunner()
    results_df = trial_runner.run_trials(
            selector=selector,
            query=query,
            sampler=sampler,
            source=source,
            nb_trials=100,
            verbose=verbose
    )
    return results_df


def run_rt(csv_fname, budget, rt, delta=0.05):
    return run_helper(csv_fname, budget, rt, 'rt', delta=delta)


def run_pt(csv_fname, budget, pt, delta=0.05):
    return run_helper(csv_fname, budget, pt, 'pt', delta=delta)
