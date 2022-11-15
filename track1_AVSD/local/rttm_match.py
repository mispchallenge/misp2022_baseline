from itertools import permutations, repeat
import numpy as np
import sys

def load_rttm(path):
    rttm = {}
    T = 120 * 60 * 100
    with open(path) as INPUT:
        for line in INPUT:
            '''
            SPEAKER session0_CH0_0S 1 417.315   9.000 <NA> <NA> 1 <NA> <NA>
            '''
            line = line.split(" ")
            while "" in line:
                line.remove("")
            session = line[1]
            if not session in rttm.keys() :
                rttm[session] = {}
            if line[-2] != "<NA>":
                spk = line[-2]
            else:
                spk = line[-3]
            if not spk in rttm[session].keys():
                rttm[session][spk] = np.zeros(T)
            #print(line[3] )
            start = np.int64(np.float64(line[3]) * 100 )
            end = start + np.int64(np.float64(line[4]) * 100)
            rttm[session][spk][start:end] = 1
    return rttm

def write_rttm(session_label, output_path):
    with open(output_path, "w") as OUT:
        for session in session_label.keys():
            for spk in session_label[session].keys():
                labels = session_label[session][spk]
                to_split = np.nonzero(labels[1:] != labels[:-1])[0]
                to_split += 1
                if labels[-1] == 1:
                    to_split = np.r_[to_split, len(labels)+1]
                if labels[0] == 1:
                    to_split = np.r_[0, to_split]
                for l in to_split.reshape(-1, 2):
                    #print(l)
                    #break
                    OUT.write("SPEAKER {} 1 {:.2f} {:.2f} <NA> <NA> {} <NA> <NA>\n".format(session, l[0]/100., (l[1]-l[0])/100., spk))


ref_rttm = load_rttm(sys.argv[1])
hyp_rttm = load_rttm(sys.argv[2])
output_rttm = sys.argv[3]

hyp_rename_rttm = {}
for session in ref_rttm.keys():
    ref_spk = list(ref_rttm[session].keys())
    hyp_spk = list(hyp_rttm[session].keys())
    ref_spk_id = list(range(len(ref_spk)))
    hyp_spk_id = list(range(len(hyp_spk)))
    if len(ref_spk) <= len(hyp_spk):
        permute = list(list(zip(r, p)) for (r, p) in zip(repeat(ref_spk_id), permutations(hyp_spk_id)))
    else:
        permute = list(list(zip(r, p)) for (r, p) in zip(permutations(ref_spk_id), repeat(hyp_spk_id)))
    score = np.zeros([len(ref_spk), len(hyp_spk)])
    for rs in ref_spk_id:
        for hs in hyp_spk_id:
            score[rs, hs] = np.sum( ref_rttm[session][ref_spk[rs]] == hyp_rttm[session][hyp_spk[hs]] )
    max_intervals = 0
    for p in permute:
        intervals = 0
        for rs, hs in p:
            intervals += score[rs, hs]
        if intervals > max_intervals:
            best_permute = p
            max_intervals = intervals
    hyp_rename_rttm[session] = {}
    print([f"{ref_spk[rs]}-{hyp_spk[hs]}" for rs, hs in  best_permute])
    for rs, hs in best_permute:
        hyp_rename_rttm[session][ref_spk[rs]] = hyp_rttm[session][hyp_spk[hs]]
        hyp_spk_id.remove(hs)
    for id, hs in enumerate(hyp_spk_id):
        hyp_rename_rttm[session][f"E{id}#{hyp_spk[hs]}"] = hyp_rttm[session][hyp_spk[hs]]

write_rttm(hyp_rename_rttm, output_rttm)
