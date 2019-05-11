from __future__ import unicode_literals, print_function, division

import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

from attndecoder import AttnDecoderRNN
from encoder import EncoderRNN
from util import prepareData, showPlot, variableFromSentence, timeSince, variablesFromPair, filter_embedding

SOS_token = 0
EOS_token = 1
teacher_forcing_ratio = 0.5
torch.set_default_tensor_type(torch.DoubleTensor)


def get_n_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def evaluate(input_lang, encoder, decoder, sentence, max_length=40):
    input_variable = variableFromSentence(input_lang, sentence)
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    #     encoder_outputs = encoder_outputs.cuda()

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    #     decoder_input = decoder_input.cuda()

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(lang.index2word[ni.item()])

        decoder_input = Variable(torch.LongTensor([[ni]]))
    #         decoder_input = decoder_input.cuda()
    return decoded_words, decoder_attentions[:di + 1]


def evaluateRandomly(lang, encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(lang, encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          max_length=40):
    encoder_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    #     encoder_outputs = encoder_outputs.cuda()
    loss = 0
    for ei in range(input_length):
        e_i = input_variable[ei]
        encoder_output, encoder_hidden = encoder(e_i, encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]
    sos_t = np.array([SOS_token])
    decoder_input = Variable(torch.LongTensor(sos_t.tolist()))
    #     decoder_input = decoder_input.cuda()

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False


    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]  # index of most voted word----------------
            dec_ni = np.array([ni])
            decoder_input = Variable(torch.LongTensor(dec_ni.tolist()))
            #             decoder_input = decoder_input.cuda()
            decoder_input = decoder_input
            if ni == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data / target_length


def trainIters(epoch, pairs, lang, encoder, decoder, print_every=100, plot_every=200, learning_rate=0.001):
    start = time.time()
    print("Starting training")
    plot_losses = []
    num = len(pairs)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for epo in range(epoch):
        print("epoch: ", epo + 1)
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every
        for iter in range(1, len(pairs) + 1):
            training_pair = variablesFromPair(pairs[iter - 1], lang)
            input_variable = training_pair[0]
            target_variable = training_pair[1]
            loss = train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer,
                         criterion)
            print("loss: ", loss.numpy())
            print_loss_total += loss
            plot_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, iter / num),
                                             iter, iter / num * 100, print_loss_avg))
                evaluateRandomly(lang, encoder, decoder, 1)

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

    showPlot(plot_losses)

    # torch.save(encoder.state_dict(), './model/encoder_model.pth')
    # torch.save(decoder.state_dict(), './model/decoder_model.pth')


lang, pairs = prepareData("data/train/article_train.txt", "data/train/title_train.txt")
filtered_embeddings = filter_embedding(lang, "glove.6B/glove.6B.100d.txt")
print(random.choice(pairs))
hidden_size = 256
encoder1 = EncoderRNN(lang.n_words, hidden_size, filtered_embeddings)
attn_decoder1 = AttnDecoderRNN(hidden_size, lang.n_words, dropout_p=0.1, embeddings=filtered_embeddings)
print("parameters ", get_n_params(encoder1) + get_n_params(attn_decoder1))
trainIters(epoch=3, pairs=pairs, lang=lang, encoder=encoder1, decoder=attn_decoder1)

_, pairs_test = prepareData("data/test/article_test.txt", "data/test/title_test.txt")


def cal_ROUGE_testset(pairs_test, print_every=200):
    rouge1 = 0
    rouge2 = 0
    for i in range(len(pairs_test)):
        predict_output, _ = evaluate(lang, encoder1, attn_decoder1, pairs_test[i][0], max_length=40)
        predict_output = ' '.join(predict_output)
        rouge1 += Rouge_1(predict_output, pairs_test[i][1])
        rouge2 += Rouge_2(predict_output, pairs_test[i][1])
        if i % print_every == 0:
            print(pairs_test[i][0] + "  ----->  " + predict_output)

    return rouge1 / len(pairs_test), rouge2 / len(pairs_test)


import jieba


# 使用jieba进行分词
def Rouge_1(model, reference):  # terms_reference为参考摘要，terms_model为候选摘要   ***one-gram*** 一元模型
    terms_reference = jieba.cut(reference)  # 默认精准模式
    terms_model = jieba.cut(model)
    grams_reference = list(terms_reference)
    grams_model = list(terms_model)
    grams_reference = [x for x in grams_reference if x != " "]
    grams_model = [x for x in grams_model if x != " "]
    temp = 0
    ngram_all = len(grams_reference)
    for x in grams_reference:
        if x in grams_model: temp = temp + 1
    rouge_1 = temp / ngram_all
    return rouge_1


def Rouge_2(model, reference):  # terms_reference为参考摘要，terms_model为候选摘要   ***Bi-gram***  2元模型
    terms_reference = jieba.cut(reference)
    terms_model = jieba.cut(model)
    grams_reference = list(terms_reference)
    grams_model = list(terms_model)
    grams_reference = [x for x in grams_reference if x != " "]
    grams_model = [x for x in grams_model if x != " "]
    gram_2_model = []
    gram_2_reference = []
    temp = 0
    ngram_all = len(grams_reference) - 1
    for x in range(len(grams_model) - 1):
        gram_2_model.append(grams_model[x] + grams_model[x + 1])
    for x in range(len(grams_reference) - 1):
        gram_2_reference.append(grams_reference[x] + grams_reference[x + 1])
    for x in gram_2_model:
        if x in gram_2_reference: temp = temp + 1
    rouge_2 = temp / ngram_all
    return rouge_2


ROUGE1, ROUGE2 = cal_ROUGE_testset(pairs_test)

print("testset: ROUGE1 ->", ROUGE1)
print("testset: ROUGE2 ->", ROUGE2)
