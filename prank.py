import numpy as np
import torch
import os
import torch.nn as nn
from torch.nn.init import xavier_normal


class PRank(nn.Module):
    def __init__(self, word_number, embed_size, bias_number):
        super(PRank,self).__init__()

        self.word_number = word_number
        self.embed_size = embed_size
        self.bias_number = bias_number - 1

        # for context words and target words
        # self.in_embed = nn.Embedding(self.word_number, self.embed_size)
        # self.in_embed.weight = xavier_normal(self.in_embed.weight)

        self.in_embed = torch.rand(self.word_number, self.embed_size, requires_grad=False)

        # self.in_bias = nn.Embedding(self.word_number, self.bias_number)
        # self.in_bias.weight[:] = 0

        self.in_bias = torch.zeros(self.word_number, self.bias_number, requires_grad=False)
        # self.in_bias.weight[:, -1] = 100000000 # set the last bias as infinte

        # for target words
        # self.out_embed = nn.Embedding(self.word_number, self.embed_size)
        # self.out_embed.weight = xavier_normal(self.out_embed.weight)

    def forward(self, context_id, target_ids, labels):
        '''
            context_id: (1)
            target_ids: (1, batch_size)
            labels: (1, batch_size)
        '''
        # context_id, target_ids, labels = input
        context_id = context_id.cuda()
        target_ids = target_ids.cuda()
        labels = labels.cuda()
        self.in_embed = self.in_embed.cuda()
        self.in_bias = self.in_bias.cuda()

        target_ids = target_ids.view(-1).long()
        labels = labels.view(-1).long()

        # context_embedding = self.in_embed(context_id) #(1, embed_size)
        context_embedding = self.in_embed[context_id] #(1, embed_size)
        # target_bias = self.in_bias(context_id) #(1, bias_number)
        target_bias = self.in_bias[context_id] #(1, bias_number)
        # target_embeddings = self.in_embed(target_ids) #(batch_size, embed_size)
        target_embeddings = self.in_embed[target_ids] #(batch_size, embed_size)
        dots = dot_product(context_embedding, target_embeddings) #(1, batch_size)

        batch_size = dots.size(1)
        temp_bias  = target_bias.repeat(batch_size, 1) #(batch_size, bias_number)
        dots = dots.squeeze(0).unsqueeze(1).repeat(1, self.bias_number) #(batch_size, bias_number)
        dots_bias = dots - temp_bias

        p_labels = self.predicted_labels(dots_bias)
        acc = self.accuracy(p_labels, labels)

        yt = self.generate_yt(batch_size, self.bias_number, labels) #(batch, bias_number)
        judge_matrix = dots_bias * yt

        tau = labels.unsqueeze(1).repeat(1, self.bias_number).float() #(batch,bias_number)
        x_co, y_co = torch.where(judge_matrix > 0)
        tau[x_co, y_co] = 0
        bias_update = tau.mean(0).unsqueeze(0)
        weight_update = (tau.sum(1).unsqueeze(1) * target_embeddings).mean(0).unsqueeze(0)
        # update
        # self.in_embed.weight[context_id] = self.in_embed.weight[context_id] + weight_update
        # self.in_bias.weight[context_id] = self.in_bias.weight[context_id] - bias_update
        self.in_embed[context_id] = self.in_embed[context_id] + weight_update
        self.in_bias[context_id] = self.in_bias[context_id] - bias_update
    
        return acc

    def generate_yt(self,batch_size, bias_number, labels):
        yt = torch.zeros(batch_size, bias_number)
        yt = yt.fill_(-1)
        # yt[:, :labels] = -1
        for i, label in enumerate(labels):
            yt[i, :label] = 1
        yt = yt.cuda()
        return yt

    def predicted_labels(self, dots_bias):
        p_labels = []
        zeros = torch.zeros(dots_bias.size(0), dots_bias.size(1))
        zeros = zeros.cuda()
        temp_dots = torch.where(dots_bias < 0, zeros, dots_bias)
        batch_size = temp_dots.size(0)
        for i in range(batch_size):
            flag = -1
            for j in range(self.bias_number):
                if temp_dots[i][j] == 0:
                    flag = j
                    break
            if flag != -1:
                p_labels.append(flag + 1)
            else:
                p_labels.append(self.bias_number + 1)
        p_labels = torch.tensor(p_labels)
        p_labels = p_labels.cuda()
        return p_labels

    def accuracy(self, p_labels, labels):
        batch_size = p_labels.size(0)
        p_labels = p_labels.cpu().numpy()
        labels = labels.cpu().numpy()
        correct = np.sum(p_labels == labels)
        acc = correct * 1.0 / batch_size
        return acc

    def get_embeddings(self):
        # return self.in_embed.weight.data.cpu().numpy() + self.out_embed.weight.data.cpu().numpy()
        # return self.in_embed.weight.data.cpu().numpy()
        return self.in_embed.data.cpu().numpy()

def dot_product(x, y):
    # x: N * D
    # y: M * D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)


    yT = torch.transpose(y, 1, 0)

    output = x @ yT

    return output
