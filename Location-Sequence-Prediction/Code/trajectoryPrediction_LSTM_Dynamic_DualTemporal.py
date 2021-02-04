# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import math
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable

#--choose device--#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

SEED = 2
random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)

######################################################################
SOS_token = 0
EOS_token = 1
global MAX_LENGTH
MAX_LENGTH = 0

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.driverID2index = {}
        self.n_drivers = 0
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        if 'Attr' in sentence:
            tokens = sentence.split('Attr')[0].split(',')
            self.addAttr(sentence.split('Attr')[1])
        else:
            tokens = sentence.split(',')
        global MAX_LENGTH
        MAX_LENGTH = max(len(tokens) - 1, MAX_LENGTH)
        for i in range(0, len(tokens) - 1):
            self.addWord(tokens[i])

    def addAttr(self, word):
        attr_tokens = word.split(',')
        if attr_tokens[1] not in self.driverID2index:
            self.driverID2index[attr_tokens[1]] = self.n_drivers
            self.n_drivers += 1

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


######################################################################
def readLangs(lang1, lang2, dir):
    #--lang1 and lang2 are respectively, the name of first, and second sequeze--#
    print("Reading lines...")
    OD_dir = dir
    #--Read the file and split into lines--#
    lines = open(OD_dir, encoding='utf-8'). \
        read().strip().split('\n')

    #--Split every line into pairs and normalize--#
    pairs = [[s for s in l.split('#')] for l in lines]
    # Reverse pairs, make Lang instances
    input_lang = Lang(lang1)
    output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


######################################################################
def prepareData(lang1, lang2, dir):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, dir)
    print("Read %s week-pairs" % len(pairs))
    print("Counting grids...")
    for pair in pairs:
        for i in range(len(pair)):
            input_lang.addSentence(pair[i])
        output_lang.addSentence(pair[len(pair) - 1])
    print("Counted grids:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


#--Please change sample data to correct dir--#
all_dir = r'D:\whu\neurocomputing\data_para\data/981762.txt'
input_lang, output_lang, all_pairs = prepareData('firstWeek', 'secondWeek', all_dir)

"""
train_pairs = all_pairs[0:int(len(all_pairs) * 0.90)]
test_pairs = all_pairs[int(len(all_pairs) * 0.90):len(all_pairs)]
"""
train_pairs =random.sample(all_pairs,int(len(all_pairs) * 0.90))
test_pairs=[]
for item in all_pairs:
    if item not in train_pairs:
        test_pairs.append(item)
print(random.choice(train_pairs))
MAX_LENGTH += 1
MAX_LENGTH *= 7
print('MAX_LENGTH:%s' % MAX_LENGTH)

######################################################################
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers=num_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.weekdayEmb = nn.Embedding(7, 10)
        self.deltadayEmb = nn.Embedding(14, 10)
        self.lstm = nn.LSTM(hidden_size + 20, hidden_size,num_layers,batch_first=True)

    def forward(self, input, attr, hidden, cell):
        embedded = self.embedding(input).view(1, 1, -1)
        weekday_emb = self.weekdayEmb(attr[0]).view(1, 1, -1)
        deltaday_emb = self.deltadayEmb(attr[1]).view(1, 1, -1)
        embedded_combine = torch.cat((embedded, weekday_emb, deltaday_emb), 2)

        output, (hidden,cell) = self.lstm(embedded_combine, (hidden,cell))
        return embedded_combine, output, (hidden,cell)  #--the dimension of output = 256--#

    def initHidden(self):
        h0 = torch.zeros(self.num_layers, 1, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, 1, self.hidden_size).to(device)
        return h0, c0

#########################################################################
#--The Attention-based RNN encoder--#
class AttnEncoderRNN(nn.Module):
    def __init__(self, attn_size, hidden_size):
        super(AttnEncoderRNN, self).__init__()
        self.attn_size = attn_size
        self.hidden_size = hidden_size
        self.WD = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.UD = nn.Linear(self.hidden_size, self.hidden_size)
        self.VD = nn.Linear(self.hidden_size, 1)
    def forward(self, hidden, cell, encoder_hts):
        ht_cell = torch.cat((hidden[0], cell[0]), dim=1)
        WD_ht_cell = self.WD(ht_cell).expand(self.attn_size, self.hidden_size)
        encoder_Hts = self.UD(encoder_hts)
        attn_weights = torch.transpose(self.VD(F.tanh(WD_ht_cell + encoder_Hts)), 0, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_hts.unsqueeze(0))
        return attn_applied

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

#########################################################################
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

######################################################################
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers,dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers=num_layers
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn_combine = nn.Linear(self.hidden_size * 2+20, self.hidden_size)  #--the feature num of attr is totally 30--#
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, self.num_layers,batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.WD=nn.Linear(self.hidden_size*2,self.hidden_size)
        self.UD=nn.Linear(self.hidden_size,self.hidden_size)
        self.VD=nn.Linear(self.hidden_size,1)
        self.weekdayEmb = nn.Embedding(7, 10)
        self.deltadayEmb = nn.Embedding(14, 10)
    def forward(self, input,attr, hidden,cell, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        weekday_emb = self.weekdayEmb(attr[0]).view(1, 1, -1)
        deltaday_emb = self.deltadayEmb(attr[1]).view(1, 1, -1)
        embedded_combine = torch.cat((embedded, weekday_emb, deltaday_emb), 2)
        embedded_combine = self.dropout(embedded_combine)
        ht_cell=torch.cat((hidden[0],cell[0]),dim=1)
        WD_ht_cell=self.WD(ht_cell).expand(7,self.hidden_size)
        encoder_Hts=self.UD(encoder_outputs)
        attn_weights=torch.transpose(self.VD(F.tanh(WD_ht_cell+encoder_Hts)),0,1)
        attn_weights=F.softmax(attn_weights,dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded_combine[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, (hidden,cell) = self.lstm(output, (hidden,cell))

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden,cell, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

######################################################################
def indexesFromSentence(lang, sentence):
    if 'Attr' in sentence:
        sentence0 = sentence.split('Attr')[0].split(',')
        Pts = [lang.word2index[sentence0[i]] for i in range(0, len(sentence0) - 1)]
        attr_tokens = sentence.split('Attr')[1].split(',')
        AttrIndex = []
        AttrIndex.append(int(attr_tokens[2]))
        AttrIndex.append(int(attr_tokens[3]))
        return Pts, AttrIndex
    else:
        sentence0 = sentence.split(',')
        Pts = [lang.word2index[sentence0[i]] for i in range(0, len(sentence0) - 1)]
        return Pts

def tensorFromSentence(lang, sentence):
    if 'Attr' in sentence:
        indexes, AttrIndex = indexesFromSentence(lang, sentence)
        indexes.append(EOS_token)
        return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1), torch.tensor(AttrIndex,
                                                                                                dtype=torch.long,
                                                                                                device=device)
    else:
        indexes = indexesFromSentence(lang, sentence)
        indexes.append(EOS_token)
        return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair):
    inputs = []
    for i in range(len(pair) - 1):
        input_tensor, attr_tensor = tensorFromSentence(input_lang, pair[i])
        inputs.append([input_tensor, attr_tensor])
    target_tensor = tensorFromSentence(output_lang, pair[len(pair) - 1])
    return (inputs, target_tensor)

######################################################################
teacher_forcing_ratio = 0.5
""""""
def train(input_tensors, target_tensor, encoder, attn_encoder, decoder, encoder_optimizer, encoder_attn_optimizer,
          decoder_optimizer, criterion, max_length=MAX_LENGTH):
    global MAX_LENGTH
    max_length = MAX_LENGTH
    # print("train:%s"%MAX_LENGTH)
    encoder_hidden, encoder_cell = encoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    encoder_attn_optimizer.zero_grad()
    target_length = target_tensor.size(0)
    loss = 0
    h_num=0
    target_Attr = []
    target_dayID = input_tensors[len(input_tensors) - 1][1][0].item() + input_tensors[len(input_tensors) - 1][1][1].item()
    target_dayID = target_dayID if target_dayID < 7 else target_dayID - 7
    target_Attr.append(target_dayID)
    target_Attr.append(0)
    target_Attr = torch.tensor(target_Attr, dtype=torch.long, device=device)
    encoder_length = max_length
    encoder_hts = torch.zeros(encoder_length, encoder.hidden_size, device=device)
    period_idx=[]
    for i in range(len(input_tensors)):
        input_tensor = input_tensors[i][0]
        input_tensor_Attr = input_tensors[i][1]
        input_length = input_tensor.size(0)
        for ei in range(input_length):
            embeded_combine, encoder_output, (encoder_hidden,encoder_cell) = encoder(
                input_tensor[ei], input_tensor_Attr, encoder_hidden,encoder_cell)
            encoder_hts[h_num] = encoder_output[0, 0]
            h_num += 1
        period_idx.append(h_num)


    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden,decoder_cell = encoder_hidden,encoder_cell

    # use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    use_teacher_forcing = True
    if use_teacher_forcing:
        #--Teacher forcing: Feed the target as the next input--#
        for di in range(target_length):
            pre_idx=0
            day_hts = torch.zeros(encoder_length, encoder.hidden_size, device=device)
            encoder_attn_outputs = torch.zeros(7, (encoder.hidden_size), device=device)
            for day_idx in range(len(period_idx)):
                cur_idx=period_idx[day_idx]
                day_hts[day_idx*int(encoder_length/7):day_idx*int(encoder_length/7)+(cur_idx-pre_idx)]=encoder_hts[pre_idx:cur_idx]
                temp=torch.zeros(int(encoder_length/7), encoder.hidden_size, device=device)
                temp[:]=day_hts[day_idx*int(encoder_length/7):(day_idx+1)*int(encoder_length/7)]
                attn_h = attn_encoder(decoder_hidden, decoder_cell, temp)
                #attn_h = attn_encoder(decoder_hidden, decoder_cell,encoder_hts[pre_idx:cur_idx])
                encoder_attn_outputs[day_idx] = attn_h[0, 0]
                pre_idx=period_idx[day_idx]
            decoder_output, decoder_hidden,decoder_cell, decoder_attention = decoder(
                decoder_input, target_Attr,decoder_hidden,decoder_cell, encoder_attn_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()
    # print("loss:" + str(loss.item() / target_length))
    return loss.item() / target_length


def space_loss(topi, targeti):
    predict_xy = output_lang.index2word[topi.squeeze().detach().item()]
    predict_xy = predict_xy.split('P')
    target_xy = output_lang.index2word[targeti.squeeze().detach().item()]
    target_xy = target_xy.split('P')
    delta_x = np.square(float(predict_xy[0]) - float(target_xy[0]))
    delta_y = np.square(float(predict_xy[1]) - float(target_xy[1]))

    return np.sqrt(delta_x + delta_y)


######################################################################
import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


######################################################################
def trainIters(encoder, attn_encoder, decoder, n_iters, print_every=1000, plot_every=10, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  #--Reset every print_every--#
    plot_loss_total = 0  #--Reset every plot_every--#

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate,momentum=0.7)
    encoder_attn_optimizer = optim.SGD(attn_encoder.parameters(), lr=learning_rate,momentum=0.7)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate,momentum=0.7)
    training_pairs = [tensorsFromPair(train_pairs[int(math.fmod(i, len(train_pairs)))]) for i in
                      range(int(n_iters * len(train_pairs)))]
    criterion = nn.NLLLoss()
    mean_accuracy_re_min=0.28
    time_cost = []
    for iter in range(1, int(n_iters * len(train_pairs)) + 1):
        training_pair = training_pairs[iter - 1]
        input_tensors = training_pair[0]
        # input_tensor_Attr=training_pair[1]
        target_tensor = training_pair[1]


        time_start_seq = time.time()
        loss = train(input_tensors, target_tensor, encoder, attn_encoder,
                     decoder, encoder_optimizer, encoder_attn_optimizer, decoder_optimizer, criterion)
        time_end_seq = time.time()
        time_cost.append(time_end_seq - time_start_seq)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print('average time cost per sequence:', np.mean(time_cost))
            time_cost = []
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / int(n_iters * len(train_pairs))),
                                         iter, iter / int(n_iters * len(train_pairs)) * 100, print_loss_avg))
            mean_accuracy_re, mean_accuracy, mean_recall = evaluateRandomly(encoder, attn_encoder, decoder, n=1)

            if mean_accuracy_re_min>mean_accuracy_re:
                mean_accuracy_re_min=mean_accuracy_re
                torch.save(encoder, rootdir + driverID + 'encoder_new.pth')
                torch.save(attn_encoder, rootdir + driverID + 'attn_encoder_new.pth')
                torch.save(decoder, rootdir + driverID + 'attn_decoder_new.pth')
        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)


######################################################################
import matplotlib as mp

mp.use('TkAgg')
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    #--this locator puts ticks at regular intervals--#
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.show()


######################################################################
def edit(str1, str2):
    list1=str1.split(',')
    list2=str2.split(',')
    matrix = [[i + j for j in range(len(list2) + 1)] for i in range(len(list1) + 1)]
    for i in range(1, len(list1) + 1):
        for j in range(1, len(list2) + 1):
            if list1[i - 1] == list2[j - 1]:
                d = 0
            else:
                d = 1
            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d)
    return matrix[len(list1)][len(list2)]
def evaluate_loss(encoder, attn_encoder, decoder, input_sentences, output_sentence, max_length=MAX_LENGTH, top_k=3):
    global MAX_LENGTH
    max_length = MAX_LENGTH
    criterion = nn.NLLLoss()
    input_tensors = []
    with torch.no_grad():
        for i in range(len(input_sentences)):
            input_tensor, attr_tensor = tensorFromSentence(input_lang, input_sentences[i])
            input_tensors.append([input_tensor, attr_tensor])
        target_tensor = tensorFromSentence(output_lang, output_sentence)
        encoder_attn_outputs = torch.zeros(7, (encoder.hidden_size), device=device)
        encoder_hidden, encoder_cell = encoder.initHidden()
        h_num=0
        target_Attr = []
        target_dayID = input_tensors[len(input_tensors) - 1][1][0].item() + input_tensors[len(input_tensors) - 1][1][
            1].item()
        target_dayID = target_dayID if target_dayID < 7 else target_dayID - 7
        target_Attr.append(target_dayID)
        target_Attr.append(0)
        target_Attr = torch.tensor(target_Attr, dtype=torch.long, device=device)
        period_idx = []
        encoder_length = max_length
        encoder_hts = torch.zeros(encoder_length, encoder.hidden_size, device=device)
        for i in range(len(input_tensors)):
            input_tensor = input_tensors[i][0]
            input_tensor_Attr = input_tensors[i][1]
            input_length = input_tensor.size(0)

            for ei in range(input_length):
                embeded_combine, encoder_output, (encoder_hidden, encoder_cell) = encoder(
                    input_tensor[ei], input_tensor_Attr, encoder_hidden, encoder_cell)
                encoder_hts[h_num] = encoder_output[0, 0]
                h_num += 1
            period_idx.append(h_num)


        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden,decoder_cell = encoder_hidden,encoder_cell

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, 7)

        # beamSearch
        decoded_words_BS = [[] for i in range(top_k)]
        pre_idx = 0
        for day_idx in range(len(period_idx)):
            day_hts = torch.zeros(int(encoder_length / 7), encoder.hidden_size, device=device)
            day_hts[0:(period_idx[day_idx]-pre_idx)] = encoder_hts[pre_idx:period_idx[day_idx]]
            attn_h = attn_encoder(decoder_hidden, decoder_cell, day_hts)
            encoder_attn_outputs[day_idx] = attn_h[0, 0]
            pre_idx = period_idx[day_idx]
        decoder_output, decoder_hidden,decoder_cell, decoder_attention = decoder(
            decoder_input,target_Attr, decoder_hidden,decoder_cell, encoder_attn_outputs)

        #  topv, topi = decoder_output.data.topk(1)
        topk = decoder_output.data.topk(top_k)
        samples = [[] for i in range(top_k)]
        dead_k = 0
        final_samples = []
        for index in range(top_k):
            topk_prob = topk[0][0][index]
            topk_index = int(topk[1][0][index])
            samples[index] = [[topk_index], topk_prob, 0, 0, decoder_hidden,decoder_cell, decoder_attention, encoder_attn_outputs]
        for di in range(max_length):
            decoder_attentions[di] = decoder_attention.data
            tmp = []
            for index in range(len(samples)):
                samples_tmp = []
                decoder_input = torch.tensor([[samples[index][0][-1]]], device=device)
                sequence, pre_scores, fin_scores, ave_scores, decoder_hidden,decoder_cell, decoder_attention, encoder_outputs = \
                samples[index]
                pre_idx = 0
                for day_idx in range(len(period_idx)):
                    day_hts = torch.zeros(int(encoder_length/7), encoder.hidden_size, device=device)
                    day_hts[0:(period_idx[day_idx]-pre_idx)] = encoder_hts[pre_idx:period_idx[day_idx]]
                    attn_h = attn_encoder(decoder_hidden, decoder_cell, day_hts)
                    encoder_attn_outputs[day_idx] = attn_h[0, 0]
                    pre_idx = period_idx[day_idx]
                decoder_output, decoder_hidden, decoder_cell, decoder_attention = decoder(decoder_input, target_Attr,decoder_hidden,decoder_cell,
                                                                            encoder_outputs)

                #--choose topk at every step--#
                topk = decoder_output.data.topk(top_k)
                for k in range(top_k):
                    topk_prob = topk[0][0][k]
                    topk_index = int(topk[1][0][k])
                    now_scores = pre_scores + topk_prob
                    fin_scores = now_scores
                    # fin_scores = pre_scores - (k - 1 ) * alpha
                    samples_tmp.append(
                        [sequence + [topk_index], now_scores, fin_scores, ave_scores, decoder_hidden,decoder_cell, decoder_attention,
                         encoder_outputs])
                tmp.extend(samples_tmp)

            samples = []
            #--choose topk finally--#
            df = pd.DataFrame(tmp)
            df.columns = ['sequence', 'pre_socres', 'fin_scores', "ave_scores", "decoder_hidden","decoder_cell",  "decoder_attention",
                          "encoder_outputs"]

            # sequence_len = df.sequence.apply(lambda x:len(x))
            # df['ave_scores'] = df['fin_scores'] / sequence_len
            df['ave_scores'] = df['fin_scores']

            df = df.sort_values('ave_scores', ascending=False).reset_index().drop(['index'], axis=1)
            df = df[:(top_k - dead_k)]
            for index in range(len(df)):
                group = df.loc[index]
                if group.tolist()[0][-1] == EOS_token:
                    final_samples.append(group.tolist())
                    df = df.drop([index], axis=0)
                    dead_k += 1
                # print("drop {}, {}".format(group.tolist()[0], dead_k))
            samples = df.values.tolist()
            if len(samples) == 0:
                break
        if len(final_samples) < top_k:
            final_samples.extend(samples[:(top_k - dead_k)])
        #--tensor2GPSdata and calculate space_loss--#
        for index in range(top_k):
            sample = final_samples[index]
            for i in sample[0]:
                if i == EOS_token:
                    break
                else:
                    decoded_words_BS[index].append(output_lang.index2word[i])
        target_num = len(output_sentence.split(',')) - 1
        accuracy = []
        recall_ratio = []
        F_value = []
        accuracy_re=[]
        for index in range(len(decoded_words_BS)):
            target = output_sentence[0:(len(output_sentence) - 1)]
            predict = ""
            for i in range(len(decoded_words_BS[index])):
                predict = predict + decoded_words_BS[index][i] + ','
            predict = predict[0:(len(predict) - 1)]
            # dist=1-edit(target,predict)/(max(len(target.split(',')),len(predict.split(','))))
            dist = edit(target, predict) / max(len(target.split(',')), len(predict.split(',')))
            accuracy_re.append(dist)

            temp_target = output_sentence.split(',')
            predict_words = decoded_words_BS[index]
            predict_length = len(predict_words)
            right_num = 0
            for word in predict_words:
                if word in temp_target:
                    right_num += 1
                    temp_target.remove(word)
                # predict_words.remove(word)

            if predict_length == 0:
                accuracy0 = 0
            else:
                accuracy0 = right_num / predict_length
            # accuracy0 = right_num / predict_length
            recall0 = right_num / target_num
            accuracy.append(accuracy0)
            recall_ratio.append(recall0)
            F_value.append(2 * accuracy0 * recall0 / (accuracy0 + recall0+0.000000001))
        # final_F = max(F_value)
        final_accuracy_re = min(accuracy_re)
        final_accuracy = accuracy[accuracy_re.index(final_accuracy_re)]
        final_recall_ratio = recall_ratio[accuracy_re.index(final_accuracy_re)]
        return decoded_words_BS[accuracy_re.index(final_accuracy_re)], final_accuracy_re,final_accuracy, final_recall_ratio#, final_F

        # final_accuracy = max(accuracy)
        # return decoded_words_BS[accuracy.index(final_accuracy)], final_accuracy

######################################################################
def showAttention(input_sentence, output_words, attentions):
    #--Set up figure with colorbar--#
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')

    fig.colorbar(cax)

    #--Set up axes--#
    # ax.set_xticklabels([''] + input_sentence.split('Attr')[0].split(',') +['<EOS>'], rotation=30)
    ax.set_xticklabels([''] + input_sentence.split('Attr')[0].split(','), rotation=30)
    ax.set_yticklabels([''] + output_words)

    #--Show label at every tick--#
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show()

import numpy as np
def evaluateRandomly(encoder, attn_encoder, decoder, n=1):
    plot_accuracy = []
    plot_recall = []
    plot_F = []

    for i in range(int(n * len(test_pairs))):
        pair = test_pairs[i % len(test_pairs)]
        output_words_BS, accuracy_re, accuracy, recall_ratio = evaluate_loss(encoder, attn_encoder, decoder,
                                                                       pair[0:len(pair) - 1], pair[len(pair) - 1])
        plot_accuracy.append(accuracy)
        plot_recall.append(recall_ratio)
        plot_F.append(accuracy_re)
        # output_sentence = ' '.join(output_words_BS)
        # print('target=' + pair[len(pair) - 1])
        # print('||predict=>', output_words_BS)
        # print('accuracy:' + str(accuracy) + '\n recall:' + str(recall_ratio))
        #  showAttention(pair[0], output_words, attentions)
        # compute the mean, std of accuracy and recall_ratio

    accuracy_array = np.array(plot_accuracy)
    mean_accuracy = np.mean(accuracy_array)
    std_accuracy = np.std(accuracy_array)
    recall_array = np.array(plot_recall)
    mean_recall = np.mean(recall_array)
    std_recall = np.std(recall_array)
    F_array = np.array(plot_F)
    mean_accuracy_re = np.mean(F_array)
    std_accuracy_re = np.std(F_array)
    print("accuracy_mean:" + str(mean_accuracy) + '\n var_accuracy:' + str(std_accuracy))
    print("recall_mean:" + str(mean_recall) + '\n var_recall:' + str(std_recall))
    print("accuracy_re:" + str(mean_accuracy_re) + '\n var_accuracy:' + str(std_accuracy_re))
    # plt.plot(plot_accuracy, '-r')
    # plt.plot(plot_recall, '-g')
    # plt.plot(plot_F, '-b')
    # plt.show()
    return mean_accuracy_re,mean_accuracy,mean_recall
        # output_words_BS, accuracy_re,accuracy,recall = evaluate_loss(encoder,attn_encoder, decoder, pair[0:len(pair) - 1], pair[len(pair) - 1])
#         plot_accuracy.append(accuracy)
#         """
#         plot_recall.append(recall_ratio)
#         plot_F.append(F)
#         """
#         # output_sentence = ' '.join(output_words_BS)
#         print('target=' + pair[len(pair) - 1])
#         print('||predict=>', output_words_BS)
#         print('accuracy:' + str(accuracy) )
#     #  showAttention(pair[0], output_words, attentions)
#     # compute the mean, std of accuracy and recall_ratio
#
#     accuracy_array = np.array(plot_accuracy)
#     mean_accuracy = np.mean(accuracy_array)
#     std_accuracy = np.std(accuracy_array)
#
#         #save to txt
#     filename = 'result_split_iters_v3.txt'
#     with open(filename,'a') as f:
# #        f.write("driverID:" + driverID + "\n")
#         f.write("\n\naccuracy_mean:" + str(mean_accuracy) + '   var_accuracy:' + str(std_accuracy))
#     f.close
#
#     """
#     recall_array = np.array(plot_recall)
#     mean_recall = np.mean(recall_array)
#     std_recall = np.std(recall_array)
#     F_array = np.array(plot_F)
#     mean_F = np.mean(F_array)
#     std_F = np.std(F_array)
#     print("recall_mean:" + str(mean_recall) + '\n var_recall:' + str(std_recall))
#     print("F_mean:" + str(mean_F) + '\n var_accuracy:' + str(std_F))
#     plt.plot(plot_recall, '-g')
#     plt.plot(plot_F, '-b')
#     """
#     print("accuracy_mean:" + str(mean_accuracy) + '\n var_accuracy:' + str(std_accuracy))
#     plt.plot(plot_accuracy, '-r')
#     plt.show()


######################################################################
#--the start of deep learning!!!!!!!!--#
hidden_size = 256
attn_size = int(MAX_LENGTH/7)
num_layers=1
import os

#--encoder--#
rootdir='data_para/'
driverID='paras/hted/2/981762/'
#--encoder--#
if os.path.exists(rootdir+driverID+'encoder_new.pth') == False:
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size,num_layers).to(device)
else:
    encoder1 = torch.load(rootdir+driverID+'encoder_new.pth').to(device)
#--attn_encoder--#
if os.path.exists(rootdir+driverID+'attn_encoder_new.pth') == False:
    attn_encoder1 = AttnEncoderRNN(attn_size, hidden_size).to(device)
else:
    attn_encoder1 = torch.load(rootdir+driverID+'attn_encoder_new.pth').to(device)
#--attn_decoder--#
if os.path.exists(rootdir+driverID+'attn_decoder_new.pth') == False:
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, num_layers,dropout_p=0.1, max_length=MAX_LENGTH).to(device)
else:
    attn_decoder1 = torch.load(rootdir+driverID+'attn_decoder_new.pth').to(device)

n_iters = 100

trainIters(encoder1, attn_encoder1, attn_decoder1, n_iters, print_every=1000)
# torch.save(encoder1,rootdir+driverID+'encoder_new.pth')
# torch.save(attn_encoder1,rootdir+driverID+'attn_encoder_new.pth')
# torch.save(attn_decoder1,rootdir+driverID+'attn_decoder_new.pth')

mean_accuracy_re,mean_accuracy,mean_recall=evaluateRandomly(encoder1, attn_encoder1, attn_decoder1, n=1)


######################################################################
#--Visualizing Attention--#
"""
def evaluateAndShowAttention(input_sentence):
    output_words, attentions = evaluate(
        encoder1, attn_decoder1, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)
"""

"""
evaluateAndShowAttention("21322P7039,21329P7042,21329P7042,21333P7044,21333P7044,21329P7042,21327P7042,21322P7039,Attr,981762,2,5")

evaluateAndShowAttention("21322P7039,21333P7044,21333P7044,21328P7043,21327P7042,21322P7039,Attr,981762,3,3")

evaluateAndShowAttention("21322P7039,21333P7044,21333P7044,21328P7043,21327P7042,21322P7039,Attr,981762,3,4")

evaluateAndShowAttention("21322P7039,21322P7040,21330P7043,21322P7039,Attr,981762,6,1")
"""