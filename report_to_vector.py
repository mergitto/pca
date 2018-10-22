import pickle
from collections import defaultdict
import numpy as np

def to_float64(vector_list):
    return np.asarray(vector_list, dtype=np.float64)

#資格
SHIKAKU = to_float64([
    0.42440349, 0.19072372, 0.099718586, -0.77085632, 0.16929626, 0.037787568, 0.29073009, -0.60986996, -0.053409051, 0.15223014,
    -0.4713468, -0.11732099, 0.1051048, -0.018203847, -0.22121894, 0.30620456, -0.52633685, -0.021409864, -0.22447515, -0.46648845,
    0.31003597, 0.11529651, -0.088379234, -0.15124038, 0.10020568, 0.71223253, 0.16819023, 0.3386583, -0.036085963, 0.52155113,
    0.16625008, 0.29046008, 1.0637453, 0.035186503, -0.27256778, -0.28778565, -0.07470046, -0.042177122, 0.60535085, 0.23258251,
    -0.11825229, -0.33874428, 0.4396432, -0.28831384, 0.49067006, -0.12783441, -0.21125489, -0.23436777, 0.24841367, -0.11601754,
    0.069074944, -0.1832916, -0.24743652, 0.6768468, 0.39374474, -0.35317862, 0.3815991, 0.5443055, 0.20078449, 0.20941895,
    -0.14501379, -0.033239149, -0.37616226, 0.098085329, 0.4157615, 0.17935899, -0.12494193, 0.49906084, -0.67993087, 0.11831599,
    -0.69798195, 0.025505401, -0.70000339, -0.3056469, 0.13950793, -0.18078308, -0.17298366, 0.038650803, -0.032814167, -0.52125651,
    0.11812772, 0.14842924, 0.0026944645, -0.031679679, 0.43843377, -0.57849944, 0.046856761, 0.17500384, 0.24977317, 0.21578297,
    0.23719518, 0.26466942, -0.066459574, -0.20302618, 0.33222714, -0.30841351, 0.14726992, 0.067696139, -0.40575948, -0.40840694])



def load_advice():
    with open("./advice_classification_2.pickle", 'rb') as f:
        return pickle.load(f)

def dump_pickle(data, save_file_name):
    with open(save_file_name, 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    reports = load_advice()
    reports_vector = defaultdict(list)
    for report in reports.values():
        if report['evaluation'] == 'high':
            reports_vector['high'].append(report['vectorSum'])
        else:
            reports_vector['low'].append(report['vectorSum'])

    reports_vector['input_word'].append(SHIKAKU)
    reports_vector['input_word'].append(SHIKAKU)

    dump_pickle(reports_vector, 'report_vector_shikaku.pickle')


