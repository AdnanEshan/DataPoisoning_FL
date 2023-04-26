from federated_learning.utils import replace_0_with_2
from federated_learning.utils import replace_5_with_3
from federated_learning.utils import replace_1_with_9
from federated_learning.utils import replace_4_with_6
from federated_learning.utils import replace_1_with_3
from federated_learning.utils import replace_6_with_0
from federated_learning.worker_selection import PoisonerProbability
import numpy as np
from federated_learning.worker_selection import BeforeBreakpoint
from federated_learning.worker_selection import AfterBreakpoint
from server import run_exp

def replace_1_with_9_30percent(data):
    new_data = []
    for d in data:
        if d == 1 and np.random.rand() < 0.3:
            new_data.append(9)
        else:
            new_data.append(d)
    return new_data

if __name__ == '__main__':
    START_EXP_IDX = 3000
    NUM_EXP = 3
    NUM_POISONED_WORKERS = 0
    REPLACEMENT_METHOD = replace_1_with_9_30percent
    KWARGS = {
        "BeforeBreakPoint_EPOCH" : 75,
        "BeforeBreakpoint_NUM_WORKERS_PER_ROUND" : 5,
        "AfterBreakPoint_EPOCH" : 75,
        "AfterBreakpoint_NUM_WORKERS_PER_ROUND" : 5,
    }

    for experiment_id in range(START_EXP_IDX, START_EXP_IDX + NUM_EXP):
        run_exp(REPLACEMENT_METHOD, NUM_POISONED_WORKERS, KWARGS, AfterBreakpoint(), experiment_id)
