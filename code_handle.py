import numpy as np
import json
import os

input_dir = "/mnt/lustre/yankun/data/peking_esex"
output_dir = "/mnt/lustre/yankun/learning_to_rank/data"
# original files
ppi_cooccur = "ppmi_cooccur.txt"
vocab = "vocab.txt"
# output files
id2voc = "id2voc.txt"
id_cooccur = "id_cooccur.dat"
id_value = "id_value.dat"

all_ids = []
with open(os.path.join(input_dir, vocab), 'r') as f:
    lines = f.readlines()
    len_words = len(lines)
    print("a total of {} words".format(len_words))
    with open(os.path.join(output_dir, id2voc), 'w') as f2:
        for i, line in enumerate(lines):
            id = i
            line = line.rstrip('\n\r')
            word, _ = line.split(' ')
            result_line = str(id) + ' ' + word + '\n'
            f2.write(result_line)
            all_ids.append(id)
print("id2voc is done ...")

id_cooccur = os.path.join(output_dir, id_cooccur)
id_value = os.path.join(output_dir, id_value)
fp_co = np.memmap(id_cooccur, dtype = 'int32', mode = 'w+', shape = (len_words, len_words + 1))
fp_value = np.memmap(id_value, dtype = 'float32', mode = 'w+', shape = (len_words, len_words))

with open(os.path.join(input_dir, ppi_cooccur), 'r') as f:
    current_id = -1
    new_id = -1
    target_ids = []
    ppi_values = []
    length = 0
    for line in f:
        line = line.rstrip('\r\n')
        target_id, context_id, ppi_value = line.split(' ')
        target_id = int(target_id) - 1
        context_id = int(context_id) - 1
        ppi_value = float(ppi_value)

        new_id = context_id
        if current_id != new_id:
            print("{} / {}".format(new_id, len_words - 1), flush=True)
            if current_id == -1:
                current_id = new_id
            else:
                target_ids = np.array(target_ids)
                ppi_values = np.array(ppi_values)
                fp_co[current_id, :length] = target_ids
                fp_co[current_id, -1] = length
                fp_value[current_id, :length] = ppi_values
                current_id = new_id
            target_ids = []
            ppi_values = []
            length = 0
        target_ids.append(target_id)
        ppi_values.append(ppi_value)
        length += 1

    target_ids = np.array(target_ids)
    ppi_values = np.array(ppi_values)
    fp_co[current_id, :length] = target_ids
    fp_co[current_id, -1] = length
    fp_value[current_id, :length] = ppi_values

del fp_co
del fp_value
