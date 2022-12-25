"""
We define the custom query that works for all the methods.
If index is provided and repr(class) is 'seiden' then we must build additional anchors.

"""


from benchmarks.stanford.tasti.tasti.query import BaseQuery
import numpy as np



class CustomQuery(BaseQuery):

    def finish_index_building(self):
        ### only perform this operation if the index is of type seiden
        if 'EKO' in repr(self.index):
            index = self.index
            target_dnn = self.index.target_dnn_cache
            scoring_func = self.score
            index.build_additional_anchors(target_dnn, scoring_func)


    def score(self, target_dnn_output):
        return 1.0 if len(target_dnn_output) > 0 else 0.0


    def execute(self, err_tol=0.01, confidence=0.05, y=None):

        if y == None:

            self.finish_index_building()

            y_pred, y_true = self.propagate(
                self.index.target_dnn_cache,
                self.index.reps, self.index.topk_reps, self.index.topk_dists
            )
        else:
            y_pred, y_true = y

        ## convert y_true

        y_true = np.array([float(tmp) for tmp in y_true])
        return y_pred, y_true
