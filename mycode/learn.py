import time
import argparse
import numpy as np
from datetime import datetime
from joblib import Parallel, delayed

from grapher import Grapher
from temporal_walk import Temporal_Walk
from rule_learning import Rule_Learner, rules_statistics
import time

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d", default="ICEWS14", type=str)
parser.add_argument("--rule_lengths", "-l", default="3", type=int, nargs="+")
parser.add_argument("--num_walks", "-n", default="100", type=int)
parser.add_argument("--transition_distr", default="exp", type=str)
parser.add_argument("--num_processes", "-p", default=10, type=int)
parser.add_argument("--seed", "-s", default=None, type=int)
parser.add_argument("--runnr", default=0, type=int) #ADDED JULIA

parsed = vars(parser.parse_args())
start_o = time.time()
dataset = parsed["dataset"]
rule_lengths = parsed["rule_lengths"]
rule_lengths = [rule_lengths] if (type(rule_lengths) == int) else rule_lengths
num_walks = parsed["num_walks"]
transition_distr = parsed["transition_distr"]
num_processes = parsed["num_processes"]
seed = parsed["seed"]
exp_nr = parsed['runnr']

dataset_dir = "../data/" + dataset + "/"
data = Grapher(dataset_dir)
temporal_walk = Temporal_Walk(data.train_idx, data.inv_relation_id, transition_distr)
rl = Rule_Learner(temporal_walk.edges, temporal_walk.neighbors, data.id2relation, data.inv_relation_id, dataset)
all_relations = sorted(temporal_walk.edges)  # Learn for all relations


def learn_rules(i, num_relations):
    """
    Learn rules (multiprocessing possible).

    Parameters:
        i (int): process number
        num_relations (int): minimum number of relations for each process

    Returns:
        rl.rules_dict (dict): rules dictionary
    """

    if seed:
        np.random.seed(seed)

    num_rest_relations = len(all_relations) - (i + 1) * num_relations
    if num_rest_relations >= num_relations:
        relations_idx = range(i * num_relations, (i + 1) * num_relations)
    else:
        relations_idx = range(i * num_relations, len(all_relations))

    num_rules = [0]
    for k in relations_idx:
        rel = all_relations[k]
        for length in rule_lengths:
            it_start = time.time()
            for _ in range(num_walks):
                walk_successful, walk = temporal_walk.sample_walk(length + 1, rel)
                if walk_successful:
                    rl.create_rule(walk)
            if length == 3:
                for _ in range(num_walks):
                    walk_successful, walk = temporal_walk.sample_acyclic_walk(rel)
                    if walk_successful:
                        rl.create_acyclic_rule(walk)
            it_end = time.time()
            it_time = round(it_end - it_start, 6)
            num_rules.append(sum([len(v) for k, v in rl.rules_dict.items()]) // 2)
            num_new_rules = num_rules[-1] - num_rules[-2]
            print(
                "Process {0}: relation {1}/{2}, length {3}: {4} sec, {5} rules".format(
                    i,
                    k - relations_idx[0] + 1,
                    len(relations_idx),
                    length,
                    it_time,
                    num_new_rules,
                )
            )

    return rl.rules_dict


start = time.time()
num_relations = len(all_relations) // num_processes
output = Parallel(n_jobs=num_processes)(
    delayed(learn_rules)(i, num_relations) for i in range(num_processes)
)
end = time.time()

all_rules = output[0]
for i in range(1, num_processes):
    all_rules.update(output[i])

total_time = round(end - start, 6)
print("Learning finished in {} seconds.".format(total_time))

rl.rules_dict = all_rules
rl.sort_rules_dict()
# dt = datetime.now()
dt = str(exp_nr)  #dt.strftime("%d%m%y%H%M") # modified julia
rl.save_rules(dt, rule_lengths, num_walks, transition_distr, seed)
rl.save_rules_verbalized(dt, rule_lengths, num_walks, transition_distr, seed)
rules_statistics(rl.rules_dict)



end_o = time.time()
total_time_o = round(end_o- start_o, 6)  
print("Learning for dataset", dataset, " finished in {} seconds.".format(total_time_o))
with open('learning_time.txt', 'a') as f:
    f.write(dataset+' learn:\t')
    f.write(str(total_time_o))
    f.write('\n')

