import pandas as pd
import numpy as np
import yaml
from pathlib import Path

#from .evaluate import train_call

params = yaml.safe_load(open("params.yaml"))["covisitation"]
paths = yaml.safe_load(open("config/settings.yaml"))["processed_files"]

train = pd.read_parquet(paths["train"])
test = pd.read_parquet(paths["test"])

fraction_of_sessions_to_use = params["data_fraction"]

if fraction_of_sessions_to_use != 1:
    lucky_sessions_train = train.drop_duplicates(['session']).sample(frac=fraction_of_sessions_to_use, random_state=42)[
        'session']
    subset_of_train = train[train.session.isin(lucky_sessions_train)]

    lucky_sessions_test = test.drop_duplicates(['session']).sample(frac=fraction_of_sessions_to_use, random_state=42)[
        'session']
    subset_of_test = test[test.session.isin(lucky_sessions_test)]
else:
    subset_of_train = train
    subset_of_test = test

subset_of_train.index = pd.MultiIndex.from_frame(subset_of_train[['session']])
subset_of_test.index = pd.MultiIndex.from_frame(subset_of_test[['session']])

chunk_size = 30_000
min_ts = train.ts.min()
max_ts = test.ts.max()

from collections import defaultdict, Counter

next_AIDs = defaultdict(Counter)

subsets = pd.concat([subset_of_train, subset_of_test])
sessions = subsets.session.unique()

for i in range(0, sessions.shape[0], chunk_size):
    current_chunk = subsets.loc[sessions[i]:sessions[min(sessions.shape[0] - 1, i + chunk_size - 1)]].reset_index(
        drop=True)
    current_chunk = current_chunk.groupby('session', as_index=False).nth(list(range(-30, -1))).reset_index(drop=True)
    consecutive_AIDs = current_chunk.merge(current_chunk, on='session')
    consecutive_AIDs['days_elapsed'] = (consecutive_AIDs.ts_y - consecutive_AIDs.ts_x) / ( params["hours_cutoff"] * 60 * 60 * 1000)
    consecutive_AIDs = consecutive_AIDs[(consecutive_AIDs.days_elapsed >= 0) & (consecutive_AIDs.days_elapsed <= 1)]

    for aid_x, aid_y in zip(consecutive_AIDs['aid_x'], consecutive_AIDs['aid_y']):
        next_AIDs[aid_x][aid_y] += 1

del train, subset_of_train, subsets

session_types = ['clicks', 'carts', 'orders']
test_session_AIDs = test.reset_index(drop=True).groupby('session')['aid'].apply(list)
test_session_types = test.reset_index(drop=True).groupby('session')['type'].apply(list)

labels = []

no_data = 0
no_data_all_aids = 0
p_w = params["weights"]
type_weight_multipliers = {0: p_w[0], 1: p_w[1], 2: p_w[2]}
for AIDs, types in zip(test_session_AIDs, test_session_types):
    if len(AIDs) >= 20:
        weights = np.logspace(0.1, 1, len(AIDs), base=2, endpoint=True) - 1
        aids_temp = defaultdict(lambda: 0)
        for aid, w, t in zip(AIDs, weights, types):
            aids_temp[aid] += w * type_weight_multipliers[t]

        sorted_aids = [k for k, v in sorted(aids_temp.items(), key=lambda item: -item[1])]
        labels.append(sorted_aids[:20])
    else:
        AIDs = list(dict.fromkeys(AIDs[::-1]))
        AIDs_len_start = len(AIDs)

        candidates = []
        for AID in AIDs:
            if AID in next_AIDs: candidates += [aid for aid, count in next_AIDs[AID].most_common(20)]
        AIDs += [AID for AID, cnt in Counter(candidates).most_common(40) if AID not in AIDs]

        labels.append(AIDs[:20])
        if candidates == []: no_data += 1
        if AIDs_len_start == len(AIDs): no_data_all_aids += 1

# >>> outputting results to CSV

labels_as_strings = [' '.join([str(l) for l in lls]) for lls in labels]

predictions = pd.DataFrame(data={'session_type': test_session_AIDs.index, 'labels': labels_as_strings})

prediction_dfs = []

for st in session_types:
    modified_predictions = predictions.copy()
    modified_predictions.session_type = modified_predictions.session_type.astype('str') + f'_{st}'
    prediction_dfs.append(modified_predictions)

submission = pd.concat(prediction_dfs)#.reset_index(drop=True)

labels_path = yaml.safe_load(open("config/settings.yaml"))["session_files"]["test_labels"]
# submission = submission.to_string(index=False).split('\n')
# submission = [f"{ele.split()[0]},{' '.join(ele.split()[1:])}" for ele in submission]
# train_call(Path(labels_path), submission)

pred_path = yaml.safe_load(open("config/settings.yaml"))["predictions"]
submission.to_csv(pred_path["path"], index=False)

print(f'Test sessions that we did not manage to extend based on the co-visitation matrix: {no_data_all_aids}')