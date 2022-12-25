import benchmarks.stanford.tasti.tasti.query as tasti


class NightStreetAggregateQuery(tasti.AggregateQuery):
    def score(self, target_dnn_output):
        return len(target_dnn_output)


class NightStreetAveragePositionAggregateQuery(tasti.AggregateQuery):
    def __init__(self, index, im_size = 1750):
        super().__init__(index)
        self.im_size = im_size

    def score(self, target_dnn_output):
        def proc_boxes(boxes):
            avg = 0.
            if len(boxes) == 0:
                return 0.
            for box in boxes:
                x = (box.xmin + box.xmax) / 2.
                avg += x / self.im_size
            return avg / len(boxes)
        return proc_boxes(target_dnn_output)

class NightStreetLimitQuery(tasti.LimitQuery):
    def score(self, target_dnn_output):
        return len(target_dnn_output)

class NightStreetSUPGPrecisionQuery(tasti.SUPGPrecisionQuery):
    def score(self, target_dnn_output):
        return 1.0 if len(target_dnn_output) > 0 else 0.0

class NightStreetSUPGRecallQuery(tasti.SUPGRecallQuery):
    def score(self, target_dnn_output):
        return 1.0 if len(target_dnn_output) > 0 else 0.0

class NightStreetLHSPrecisionQuery(tasti.SUPGPrecisionQuery):
    def score(self, target_dnn_output):
        def proc_boxes(boxes):
            mid = 1750 / 2
            for box in boxes:
                x = (box.xmin + box.xmax) / 2.
                if x < mid:
                    return True
            return False
        return proc_boxes(target_dnn_output)

class NightStreetLHSRecallQuery(tasti.SUPGRecallQuery):
    def score(self, target_dnn_output):
        def proc_boxes(boxes):
            mid = 1750 / 2
            for box in boxes:
                x = (box.xmin + box.xmax) / 2.
                if x < mid:
                    return True
            return False
        return proc_boxes(target_dnn_output)
