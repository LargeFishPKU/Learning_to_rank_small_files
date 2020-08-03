import numpy as np
import os

dir = "/mnt/lustre/yankun/learning_to_rank/data"
id_cooccur = "id_cooccur.dat"
id2voc = "id2voc.txt"


id_cooccur = os.path.join(dir, id_cooccur)
id2voc = os.path.join(dir, id2voc)
with open(id2voc, 'r') as f:
    length = len(f.readlines())

fp = np.memmap(id_cooccur, dtype = 'int32', mode = 'r', shape = (length, length + 1))
print(len(fp))
min_n = 100000000
for i in range(len(fp)):
    length_temp = fp[i][-1]
    x = fp[i][:length_temp]
    # import pdb; pdb.set_trace()
    # print('{}'. format(length_temp))
    # if length_temp < min_n:
    #     min_n = length_temp
    # if length_temp == 0:
    #     import pdb; pdb.set_trace()

print("the least one is {}".format(min_n))
