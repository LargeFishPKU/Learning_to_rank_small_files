import numpy as np
import torch
import os
import argparse
from easydict import EasyDict
from prank import PRank
from prank_dataset import PRank_Dataset

data_dir = "/mnt/lustre/yankun/learning_to_rank/data"
id2voc = os.path.join(data_dir, "id2voc.txt")
# id_cooccur = os.path.join(data_dir, "id_cooccur.dat")
# id_value = os.path.join(data_dir, "id_value.dat")
small_cooccur_path = "/mnt/lustre/yankun/learning_to_rank/small_cooccur"


# training process of
def train_save(batch_size, embed_size, bias_number, iterations, save_embedding_path):
    # get the dataset and the number of words
    print("prepare the dataset ...")
    # dataset = PRank_Dataset(id2voc, id_cooccur, id_value, batch_size, bias_number)
    dataset = PRank_Dataset(id2voc, small_cooccur_path, batch_size, bias_number)
    word_number = dataset.get_word_number()
    print("the number of words is {}".format(word_number))
    print("done")
    # get the prank model
    print("get prank model ...", flush=True)
    model = PRank(word_number, embed_size, bias_number)
    model = model.cuda()
    print("done")
    # train the model
    print("training ...")
    dataload = torch.utils.data.DataLoader(dataset, num_workers=12, pin_memory=True)

    for i in range(iterations):
        print("{} / {}".format(i, iterations), flush=True)
        len_dataload = len(dataload)
        avg_acc = 0.0
        for j, (context_id, target_ids, labels) in enumerate(dataload):
            acc = model(context_id, target_ids, labels)
            avg_acc = avg_acc + acc
            if j % 100 == 0:
                print("sub: {} / {}, accuracy : {:.2f}%".format(j, len_dataload, avg_acc / (j + 1) * 100), flush = True)

    # save embeddings:
    print("training process is done")
    print("start saving embeddings ...")
    if not os.path.isdir(save_embedding_path):
        os.makedirs(save_embedding_path)
    embedings = model.get_embeddings()

    # save the data as the numpy form:
    file_path = os.path.join(save_embedding_path, 'embeddings.npy')
    np.save(file_path, embedings)
    # save the data as the txt form:
    file_path = os.path.join(save_embedding_path, 'embeddings.txt')
    with open(file_path, 'w') as f:
        number_word = len(embedings)
        for i in range(number_word):
            line_i = embedings[i]
            temp = ""
            for j, value in enumerate(line_i):
                if j == 0:
                    temp = temp + str(value)
                else:
                    temp = temp + ' ' + str(value)
            temp = temp + '\n'
            f.write(temp)
    print("done")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'pranking algorithm')
    parser.add_argument('--batch_size', default = 200)
    parser.add_argument('--embed_size', default = 300)
    parser.add_argument('--bias_number', default = 5)
    parser.add_argument('--iterations', default = 5000)
    parser.add_argument('--save_embedding_path', default = "/mnt/lustre/yankun/learning_to_rank/embedding")

    args = parser.parse_args()

    # params = EasyDict(args)
    train_save(args.batch_size, args.embed_size, args.bias_number, args.iterations, args.save_embedding_path)
    # import pdb; pdb.set_trace()
