"""Run the identical canonicalization pipeline on overlapping-token coding."""

import numpy as np

from analyze_shared_pair_count import run as analyze
from shared_pair_gauge import run as gauge
from shared_pair_graph_prune import run as prune
from shared_pair_node_interventions import run as interventions
from train_shared_pair_dense import TOKEN_SIGNS,examples_dense


def run():
    truth=TOKEN_SIGNS.T/np.linalg.norm(TOKEN_SIGNS.T,axis=1,keepdims=True)
    analyze(example_source=examples_dense,model_prefix='shared_pair_dense',
        output_name='shared_pair_dense_analysis.json',
        task_description='overlapping three-bit token code plus two quadratic parity rules')
    gauge(source_name='shared_pair_dense_analysis.json',
          output_name='shared_pair_dense_gauge.json',truth=truth)
    prune(analysis_name='shared_pair_dense_analysis.json',
          gauge_name='shared_pair_dense_gauge.json',example_source=examples_dense,
          model_prefix='shared_pair_dense',output_name='shared_pair_dense_graph_prune.json')
    interventions(analysis_name='shared_pair_dense_analysis.json',
                  gauge_name='shared_pair_dense_gauge.json',
                  example_source=examples_dense,
                  output_name='shared_pair_dense_node_interventions.json')


if __name__=='__main__':run()
