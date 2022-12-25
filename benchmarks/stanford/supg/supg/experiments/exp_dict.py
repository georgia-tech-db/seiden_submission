from supg.selector import ApproxQuery


# Jackson experiments
jackson_experiments = {
    'jackson_precision_uniform': {
        'source': 'jackson',
        'sampler': 'NaiveSampler',
        'estimator': 'NaiveEstimator',
        'query': ApproxQuery(
            qtype='pt',
            min_recall=0.95, min_precision=0.95, delta=0.05,
            budget=10000),
        'selector': 'UniformPrecisionSelector',
        'num_trials': 100,
    },
    'jackson_precision_imp': {
        'source': 'jackson',
        'sampler': 'ImportanceSampler',
        'estimator': 'NaiveEstimator',
        'query': ApproxQuery(
            qtype='pt',
            min_recall=0.95, min_precision=0.95, delta=0.05,
            budget=10000),
        'selector': 'ImportancePrecisionSelector',
        'num_trials': 100,
    },
    'jackson_precision_imp_two': {
        'source': 'jackson',
        'sampler': 'ImportanceSampler',
        'estimator': 'None',
        'query': ApproxQuery(
            qtype='pt',
            min_recall=0.95, min_precision=0.95, delta=0.05,
            budget=10000),
        'selector': 'ImportancePrecisionTwoStageSelector',
        'num_trials': 100,
    },
    "jackson_recall_uniform": {
        "source": "jackson",
        'sampler': 'NaiveSampler',
        'estimator': 'NaiveEstimator',
        "query": ApproxQuery(
            qtype="rt",
            min_recall=0.9, min_precision=0.95, delta=0.05,
            budget=10000),
        "selector": "UniformRecall",
        "num_trials": 100,
    },
    "jackson_recall_imp": {
        "source": "jackson",
        "sampler": "ImportanceSampler",
        'estimator': 'None',
        "query": ApproxQuery(
            qtype="rt",
            min_recall=0.9, min_precision=0.95, delta=0.05,
            budget=10000),
        "selector": "ImportanceRecall",
        "num_trials": 100,
    },
}


# Beta experiments
beta_experiments = {
    'beta_precision_naive': {
        'source': 'beta',
        'sampler': 'NaiveSampler',
        'estimator': 'None',
        'query': ApproxQuery(
            qtype="rt",
            min_recall=0.95, min_precision=0.95, delta=0.05,
            budget=10000,
        ),
        'selector': 'NaivePrecisionSelector',
        'num_trials': 100
    },
    'beta_precision_uniform': {
        'source': 'beta',
        'sampler': 'NaiveSampler',
        'estimator': 'None',
        'query': ApproxQuery(
            qtype="rt",
            min_recall=0.95, min_precision=0.95, delta=0.05,
            budget=10000,
        ),
        'selector': 'UniformPrecisionSelector',
        'num_trials': 100
    },
    'beta_precision_imp': {
        'source': 'beta',
        'sampler': 'ImportanceSampler',
        'estimator': 'None',
        'query': ApproxQuery(
            qtype="rt",
            min_recall=0.95, min_precision=0.95, delta=0.05,
            budget=10000,
        ),
        'selector': 'ImportancePrecisionSelector',
        'num_trials': 100
    },
    'beta_precision_imp_two': {
        'source': 'beta',
        'sampler': 'ImportanceSampler',
        'estimator': 'None',
        'query': ApproxQuery(
            qtype="rt",
            min_recall=0.95, min_precision=0.95, delta=0.05,
            budget=10000,
        ),
        'selector': 'ImportancePrecisionTwoStageSelector',
        'num_trials': 100
    },

    'beta_recall_naive': {
        'source': 'beta',
        'sampler': 'NaiveSampler',
        'estimator': 'None',
        'query': ApproxQuery(
            qtype="rt",
            min_recall=0.9, min_precision=0.95, delta=0.05,
            budget=10000,
        ),
        'selector': 'NaiveRecallSelector',
        'num_trials': 100
    },
    'beta_recall_uniform': {
        'source': 'beta',
        'sampler': 'NaiveSampler',
        'estimator': 'None',
        'query': ApproxQuery(
            qtype="rt",
            min_recall=0.9, min_precision=0.95, delta=0.05,
            budget=10000,
        ),
        'selector': 'UniformRecall',
        'num_trials': 100
    },
    'beta_recall_imp': {
        'source': 'beta',
        'sampler': 'ImportanceSampler',
        'estimator': 'None',
        'query': ApproxQuery(
            qtype="rt",
            min_recall=0.9, min_precision=0.95, delta=0.05,
            budget=1000,
        ),
        'selector': 'ImportanceRecall',
        'num_trials': 100
    },

    'beta_jt_imp': {
        'source': 'beta',
        'sampler': 'ImportanceSampler',
        'estimator': 'None',
        'query': ApproxQuery(
            qtype="jt",
            min_recall=0.9, min_precision=0.9, delta=0.05,
            budget=10000,
        ),
        'selector': 'ImportanceJoint',
        'num_trials': 100
    },
    'beta_jt_uniform': {
        'source': 'beta',
        'sampler': 'NaiveSampler',
        'estimator': 'None',
        'query': ApproxQuery(
            qtype="jt",
            min_recall=0.9, min_precision=0.9, delta=0.05,
            budget=10000,
        ),
        'selector': 'UniformJoint',
        'num_trials': 100
    },
}


# ImageNet experiments
imagenet_experiments = {
    'imagenet_precision_uniform': {
        'source': 'imagenet',
        'sampler': 'NaiveSampler',
        'estimator': 'None',
        'query': ApproxQuery(
            qtype="pt",
            min_recall=0.95, min_precision=0.95, delta=0.05,
            budget=1000,
        ),
        'selector': 'UniformPrecisionSelector',
        'num_trials': 100
    },
    'imagenet_precision_imp': {
        'source': 'imagenet',
        'sampler': 'ImportanceSampler',
        'estimator': 'None',
        'query': ApproxQuery(
            qtype="pt",
            min_recall=0.95, min_precision=0.95, delta=0.05,
            budget=1000,
        ),
        'selector': 'ImportancePrecisionSelector',
        'num_trials': 100
    },

    'imagenet_recall_uniform': {
        'source': 'imagenet',
        'sampler': 'NaiveSampler',
        'estimator': 'None',
        'query': ApproxQuery(
            qtype="rt",
            min_recall=0.9, min_precision=0.95, delta=0.05,
            budget=1000,
        ),
        'selector': 'UniformRecall',
        'num_trials': 100
    },
    'imagenet_recall_imp': {
        'source': 'imagenet',
        'sampler': 'ImportanceSampler',
        'estimator': 'None',
        'query': ApproxQuery(
            qtype="rt",
            min_recall=0.9, min_precision=0.95, delta=0.05,
            budget=1000,
        ),
        'selector': 'ImportanceRecall',
        'num_trials': 100
    },
}


onto_experiments = {
    'onto_precision_uniform': {
        'source': 'onto',
        'sampler': 'NaiveSampler',
        'estimator': 'None',
        'query': ApproxQuery(
            qtype="pt",
            min_recall=0.95, min_precision=0.95, delta=0.05,
            budget=1000,
        ),
        'selector': 'UniformPrecisionSelector',
        'num_trials': 100
    },
    'onto_precision_imp': {
        'source': 'onto',
        'sampler': 'ImportanceSampler',
        'estimator': 'None',
        'query': ApproxQuery(
            qtype="pt",
            min_recall=0.95, min_precision=0.95, delta=0.05,
            budget=1000,
        ),
        'selector': 'ImportancePrecisionSelector',
        'num_trials': 100
    },
    'onto_recall_uniform': {
        'source': 'onto',
        'sampler': 'NaiveSampler',
        'estimator': 'None',
        'query': ApproxQuery(
            qtype="rt",
            min_recall=0.9, min_precision=0.95, delta=0.05,
            budget=1000,
        ),
        'selector': 'UniformRecall',
        'num_trials': 100
    },
    'onto_recall_imp': {
        'source': 'onto',
        'sampler': 'ImportanceSampler',
        'estimator': 'None',
        'query': ApproxQuery(
            qtype="rt",
            min_recall=0.9, min_precision=0.95, delta=0.05,
            budget=1000,
        ),
        'selector': 'ImportanceRecall',
        'num_trials': 100
    },
}


tacred_experiments = {
    'tacred_precision_uniform': {
        'source': 'tacred',
        'sampler': 'NaiveSampler',
        'estimator': 'None',
        'query': ApproxQuery(
            qtype="pt",
            min_recall=0.95, min_precision=0.95, delta=0.05,
            budget=1000,
        ),
        'selector': 'UniformPrecisionSelector',
        'num_trials': 100
    },
    'tacred_precision_imp': {
        'source': 'tacred',
        'sampler': 'ImportanceSampler',
        'estimator': 'None',
        'query': ApproxQuery(
            qtype="pt",
            min_recall=0.95, min_precision=0.95, delta=0.05,
            budget=1000,
        ),
        'selector': 'ImportancePrecisionSelector',
        'num_trials': 100
    },
    'tacred_recall_uniform': {
        'source': 'tacred',
        'sampler': 'NaiveSampler',
        'estimator': 'None',
        'query': ApproxQuery(
            qtype="rt",
            min_recall=0.9, min_precision=0.95, delta=0.05,
            budget=1000,
        ),
        'selector': 'UniformRecall',
        'num_trials': 100
    },
    'tacred_recall_imp': {
        'source': 'tacred',
        'sampler': 'ImportanceSampler',
        'estimator': 'None',
        'query': ApproxQuery(
            qtype="rt",
            min_recall=0.9, min_precision=0.95, delta=0.05,
            budget=1000,
        ),
        'selector': 'ImportanceRecall',
        'num_trials': 100
    },
}


experiments = dict(
        jackson_experiments,
        **beta_experiments,
        **imagenet_experiments,
        **onto_experiments,
        **tacred_experiments,
)
