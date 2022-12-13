import pandas as pd
import argparse
import logging
from tqdm.auto import tqdm
import yaml

session_files = yaml.safe_load(open("config/settings.yaml"))["session_files"]
processed_files = yaml.safe_load(open("config/settings.yaml"))["processed_files"]

sample_size = 100_000


def process_data(chunks, out_path):
    test_df = pd.DataFrame()

    for chunk in tqdm(chunks, f"creating {out_path}"):
        event_dict = {'session': [], 'aid': [], 'ts': [], 'type': []}

        for session, events in zip(chunk['session'].tolist(), chunk['events'].tolist()):
            for event in events:
                event_dict['session'].append(session)
                event_dict['aid'].append(event['aid'])
                event_dict['ts'].append(event['ts'])
                event_dict['type'].append(event['type'])
        test_df = pd.concat([test_df, pd.DataFrame(event_dict)])

    test_df = test_df.reset_index(drop=True)
    test_df = test_df.sort_values(["session", "ts"], ascending=[True, True])
    test_df.type = test_df.type.map({"clicks": 0, "carts": 1, "orders": 2})
    test_df.to_parquet(out_path)


def main():
    chunks = pd.read_json(session_files["train"], lines=True, chunksize=sample_size)
    process_data(chunks, processed_files["train"])
    chunks = pd.read_json(session_files["test"], lines=True, chunksize=sample_size)
    process_data(chunks, processed_files["test"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main()