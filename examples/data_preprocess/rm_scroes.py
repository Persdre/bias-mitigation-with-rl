import json

input_file = '/shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_train_system_100k_no_metadata.jsonl'
output_file = '/shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_train_system_100k_no_metadata_cleaned.jsonl'

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        data = json.loads(line)
        if 'reference' in data and 'scores' in data['reference']:
            del data['reference']['scores']
        outfile.write(json.dumps(data) + '\n')