import os
import subprocess

import pandas as pd

filename = 'data/train-00000-of-00010-0e8812f0a3785769.parquet'
df = pd.read_parquet(filename)
df.rename({'content': 'input'}, axis=1, inplace=True)
df.drop(['path', 'repo_name'], axis=1, inplace=True)

# truncate to 200 characters
# df['input'] = df['input'].str[:200]

# Replace newlines with <EOL> token
df['input'] = df['input'].str.replace(r'[\n]', ' <EOL> ')

# Add a space before and after the punctuation
# df['input'] = df['input'].str.replace(r'([.,!?();{}])', r' \1 ')

df['input'] = '<s> ' + df['input'] + ' </s>'

df['id'] = df.index + 1
df = df[['id', 'input']]

if not os.path.exists('extracted'):
    os.mkdir('extracted')

output_json_filename = 'extracted/train.json'
df.to_json(output_json_filename, orient='records', lines=True)

# Replace escaped slashes with forward slashes using subprocess sed
subprocess.call(['sed', '-i', '-e', 's/\\\\\//\//g', output_json_filename])
os.remove(output_json_filename + '-e')