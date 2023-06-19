import json
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-t", default=None, type=str, help="File to gather context")
parser.add_argument("-o", default=None, type=str, help="File to output context corpus")


args = parser.parse_args()

with open(args.t, "r", encoding='utf-8') as f:
    reader_list = list(f)

input_data = []
for i, reader in enumerate(reader_list):
    if i == 0: continue
    input_data.append(json.loads(reader))

i = 0
with open(args.o, 'w', encoding='utf-8') as f:
    for corpa in input_data:
        context = corpa['context'].lower()
        f.write(context + '\n')
        i += 1

        print(i)

print("Done")
