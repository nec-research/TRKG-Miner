import json
import time
import argparse
import itertools
import numpy as np
from joblib import Parallel, delayed

import rule_application as ra
from grapher import Grapher
from temporal_walk import store_edges
from rule_learning import rules_statistics
from score_functions import score_12


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d", default="ICEWS14", type=str)
parser.add_argument("--test_data", default="test", type=str)
parser.add_argument("--rules", "-r", default="0112231719_r[3]_n100_exp_sNone_rules.json", type=str)
parser.add_argument("--rule_lengths", "-l", default=3, type=int, nargs="+")
parser.add_argument("--window", "-w", default=200, type=int)
parser.add_argument("--top_k", default=20, type=int)
parser.add_argument("--num_processes", "-p", default=10, type=int)
parser.add_argument("--runnr", default=0, type=int) #ADDED JULIA
parser.add_argument("--seed", default=0, type=int) #ADDED JULIA
parsed = vars(parser.parse_args())
start_o = time.time()
dataset = parsed["dataset"]
rules_file = parsed["rules"]
window = parsed["window"]
top_k = parsed["top_k"]
num_processes = parsed["num_processes"]
rule_lengths = parsed["rule_lengths"]
rule_lengths = [rule_lengths] if (type(rule_lengths) == int) else rule_lengths

dataset_dir = "../data/" + dataset + "/"
dir_path = "../output/" + dataset + "/"
data = Grapher(dataset_dir)
test_data = data.test_idx if (parsed["test_data"] == "test") else data.valid_idx
rules_dict = json.load(open(dir_path + rules_file))
rules_dict = {int(k): v for k, v in rules_dict.items()}
print("Rules statistics:")
rules_statistics(rules_dict)
rules_dict = ra.filter_rules(
    rules_dict, min_conf=0.01, min_body_supp=2, rule_lengths=rule_lengths
)
print("Rules statistics after pruning:")
rules_statistics(rules_dict)
learn_edges = store_edges(data.train_idx)

score_func = score_12
# It is possible to specify a list of list of arguments for tuning
args = [[0.1, 0.5]]

### added eval_paper_authors for logging 
exp_nr = 0
if window < 0:
    steps = 'multistep'
    windowname = 'minus'+str(window)
else:
    steps ='singlestep'
    windowname = str(window)
method = 'tkgr'
filter = 'timeaware'
logname = method + '-' + dataset + '-' +str(exp_nr) + '-' +steps + '-' + windowname + '-' + str(0)
print(logname)
    #renet-ICEWS18-multistep-raw-modifiedpredict_xxx_1_3206_6624.pt

##end eval_paper_authors

def apply_rules(i, num_queries):
    """
    Apply rules (multiprocessing possible).

    Parameters:
        i (int): process number
        num_queries (int): minimum number of queries for each process

    Returns:
        all_candidates (list): answer candidates with corresponding confidence scores
        no_cands_counter (int): number of queries with no answer candidates
    """

    print("Start process", i, "...")
    all_candidates = [dict() for _ in range(len(args))]
    no_cands_counter = 0

    num_rest_queries = len(test_data) - (i + 1) * num_queries
    if num_rest_queries >= num_queries:
        test_queries_idx = range(i * num_queries, (i + 1) * num_queries)
    else:
        test_queries_idx = range(i * num_queries, len(test_data))

    cur_ts = test_data[test_queries_idx[0]][3]
    first_test_query_ts = test_data[0][3] #added eval_paper_authors: first_test_query_ts
    edges = ra.get_window_edges(data.all_idx, cur_ts, learn_edges, window)

    it_start = time.time()
    eval_paper_authors_logging_dict = {} #added eval_paper_authors for logging
    for j in test_queries_idx:
        test_query = test_data[j]
        cands_dict = [dict() for _ in range(len(args))]

        if test_query[3] != cur_ts:
            cur_ts = test_query[3]
            edges = ra.get_window_edges(data.all_idx, cur_ts, learn_edges, window, first_test_query_ts) #added eval_paper_authors: first_test_query_ts

        if test_query[1] in rules_dict:
            dicts_idx = list(range(len(args)))
            for rule in rules_dict[test_query[1]]:
                if "rule_type" in rule and rule["rule_type"] == "acyclic":
                    walk_edges = ra.match_acyclic_body_relations(rule, edges, test_query[0])
                else:
                    walk_edges = ra.match_body_relations(rule, edges, test_query[0])

                if 0 not in [len(x) for x in walk_edges]:
                    if "rule_type" in rule and rule["rule_type"] == "acyclic":
                        rule_walks = ra.get_acyclic_walks(rule, walk_edges)
                    else:
                        rule_walks = ra.get_walks(rule, walk_edges)
                    if rule["var_constraints"]:
                        rule_walks = ra.check_var_constraints(
                            rule["var_constraints"], rule_walks
                        )

                    if not rule_walks.empty:
                        cands_dict = ra.get_candidates(
                            rule,
                            rule_walks,
                            cur_ts,
                            cands_dict,
                            score_func,
                            args,
                            dicts_idx,
                        )
                        for s in dicts_idx:
                            cands_dict[s] = {
                                x: sorted(cands_dict[s][x], reverse=True)
                                for x in cands_dict[s].keys()
                            }
                            cands_dict[s] = dict(
                                sorted(
                                    cands_dict[s].items(),
                                    key=lambda item: item[1],
                                    reverse=True,
                                )
                            )
                            top_k_scores = [v for _, v in cands_dict[s].items()][:top_k]
                            unique_scores = list(
                                scores for scores, _ in itertools.groupby(top_k_scores)
                            )
                            if len(unique_scores) >= top_k:
                                dicts_idx.remove(s)
                        if not dicts_idx:
                            break

            if cands_dict[0]:
                for s in range(len(args)):
                    # Calculate noisy-or scores
                    scores = list(
                        map(
                            lambda x: 1 - np.product(1 - np.array(x)),
                            cands_dict[s].values(),
                        )
                    )
                    cands_scores = dict(zip(cands_dict[s].keys(), scores))
                    noisy_or_cands = dict(
                        sorted(cands_scores.items(), key=lambda x: x[1], reverse=True)
                    )
                    all_candidates[s][j] = noisy_or_cands
            else:  # No candidates found by applying rules
                no_cands_counter += 1
                for s in range(len(args)):
                    all_candidates[s][j] = dict()

        else:  # No rules exist for this relation
            no_cands_counter += 1
            for s in range(len(args)):
                all_candidates[s][j] = dict()
        #added eval_paper_authors
        import inspect
        import sys
        import os
        currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        # parentdir = os.path.dirname(currentdir)
        sys.path.insert(1, currentdir) 
        sys.path.insert(1, os.path.join(sys.path[0], '../..'))        
   
        import evaluation_utils 
        num_nodes, num_rels = evaluation_utils.get_total_number(dataset_dir, 'stat.txt')
        query_name, gt_test_query_ids = evaluation_utils.query_name_from_quadruple(test_query, num_rels)

        predictions = evaluation_utils.create_scores_tensor(all_candidates[s][j], num_nodes)  
        eval_paper_authors_logging_dict[query_name] = [predictions, gt_test_query_ids]               
        #end eval_paper_authors t

        if not (j - test_queries_idx[0] + 1) % 100:
            it_end = time.time()
            it_time = round(it_end - it_start, 6)
            print(
                "Process {0}: test samples finished: {1}/{2}, {3} sec".format(
                    i, j - test_queries_idx[0] + 1, len(test_queries_idx), it_time
                )
            )
            it_start = time.time()

    return all_candidates, no_cands_counter, eval_paper_authors_logging_dict #eval_paper_authors added eval_paper_authors_logging_dict


start = time.time()
num_queries = len(test_data) // num_processes
output = Parallel(n_jobs=num_processes)(
    delayed(apply_rules)(i, num_queries) for i in range(num_processes)
)
end = time.time()

#added eval_paper_authors for logging
eval_paper_authors_final_logging_dict = {}
for proc_loop in range(num_processes):
    eval_paper_authors_final_logging_dict.update(output[proc_loop][2])
import pathlib
import pickle
import os
dirname = os.path.join(pathlib.Path().resolve(), 'results' )
eval_paper_authorsfilename = os.path.join(dirname, logname + ".pkl")
# if not os.path.isfile(eval_paper_authorsfilename):
with open(eval_paper_authorsfilename,'wb') as file:
    pickle.dump(eval_paper_authors_final_logging_dict, file, protocol=4) 
file.close()
#END eval_paper_authors

final_all_candidates = [dict() for _ in range(len(args))]
for s in range(len(args)):
    for i in range(num_processes):
        final_all_candidates[s].update(output[i][0][s])
        output[i][0][s].clear()

final_no_cands_counter = 0
for i in range(num_processes):
    final_no_cands_counter += output[i][1]

total_time = round(end - start, 6)
print("Application finished in {} seconds.".format(total_time))
print("No candidates: ", final_no_cands_counter, " queries")

for s in range(len(args)):
    score_func_str = score_func.__name__ + str(args[s])
    score_func_str = score_func_str.replace(" ", "")
    ra.save_candidates(
        rules_file,
        dir_path,
        final_all_candidates[s],
        rule_lengths,
        window,
        score_func_str,
    )



end_o = time.time()
total_time_o = round(end_o- start_o, 6)  
print("Apply for dataset", dataset, " finished in {} seconds.".format(total_time_o))
with open('learning_time.txt', 'a') as f:
    f.write(dataset+'apply:\t')
    f.write(str(total_time_o))
    f.write('\n')

