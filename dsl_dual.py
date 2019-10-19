# -*- coding: utf-8 -*-
import sys
import torch
import argparse
import random, os
import numpy as np

from torch.autograd import Variable
import torch.nn.functional as F

from nmt import read_corpus_for_dsl, data_iter, get_new_batch, data_iter_for_dual
from nmt import NMT, to_input_variable, decode, get_bleu, read_corpus, vocab

from lm import LMProb
from lm import model

def loss_att(a, b, mask, length):
    epsilon = 1e-8
    a.data.masked_fill_(mask, -(1e+8))
    b.data.masked_fill_(mask, -(1e+8))
    a = F.softmax(a, 2) + epsilon
    b = F.softmax(b, 2) + epsilon
    x_a = a * torch.log(a / ((b + a) / 2))
    x_b = b * torch.log(b / ((b + a) / 2))
    x_a.data.masked_fill_(mask, 0)
    x_b.data.masked_fill_(mask, 0)
    x_a = torch.sum(x_a, 2)
    x_b = torch.sum(x_b, 2)
    x_a = torch.sum(x_a, 1) / Variable(torch.FloatTensor(length).cuda())
    x_b = torch.sum(x_b, 1) / Variable(torch.FloatTensor(length).cuda())
    kl_div = x_a + x_b
    kl_div = kl_div / 2
    return kl_div.cpu()

def dual(args):
    vocabs = {}
    opts = {}
    state_dicts = {}
    train_srcs = {}
    train_tgt = {}
    lm_scores = {}
    dev_data = {}

    # load model params & training data
    for i in range(2):
        model_id = (['A', 'B'])[i]
        print('loading pieces, part {:s}'.format(model_id))
        print('  load model{:s}     from [{:s}]'.format(model_id, args.nmt[i]))
        params = torch.load(args.nmt[i], map_location=lambda storage, loc: storage)  # load model onto CPU
        vocabs[model_id] = params['vocab']
        print ('=='*10)
        print (vocabs[model_id])
        opts[model_id] = params['args']
        state_dicts[model_id] = params['state_dict']
        print ('done')

    for i in range(2):
        model_id = (['A', 'B'])[i]
        print('  load train_src{:s} from [{:s}]'.format(model_id, args.src[i]))
        train_srcs[model_id], lm_scores[model_id] = read_corpus_for_dsl(args.src[i], source='src')
        train_tgt[model_id], _ = read_corpus_for_dsl(args.src[(i+1)%2], source='tgt')


    dev_data_src1 = read_corpus(args.val[0], source='src')
    dev_data_tgt1 = read_corpus(args.val[1], source='tgt')
    dev_data['A'] = list(zip(dev_data_src1, dev_data_tgt1))
    dev_data_src2 = read_corpus(args.val[1], source='src')
    dev_data_tgt2 = read_corpus(args.val[0], source='tgt')
    dev_data['B'] = list(zip(dev_data_src2, dev_data_tgt2))
    
    models = {}
    optimizers = {}
    nll_loss = {}
    cross_entropy_loss = {}

    for m in ['A', 'B']:
        # build model
        opts[m].cuda = args.cuda

        models[m] = NMT(opts[m], vocabs[m])
        models[m].load_state_dict(state_dicts[m])
        models[m].train()

        if args.cuda:
            if m == 'A':
                models[m] = models[m].cuda()
            else:
                models[m] = models[m].cuda()

        optimizers[m] = torch.optim.SGD(filter(lambda p: p.requires_grad, models[m].parameters()), lr=args.lr)
    for m in ['A', 'B']:
        vocab_mask = torch.ones(len(vocabs[m].tgt))
        vocab_mask[vocabs[m].tgt['<pad>']] = 0
        nll_loss[m] = torch.nn.NLLLoss(weight=vocab_mask, size_average=False)
        cross_entropy_loss[m] = torch.nn.CrossEntropyLoss(weight=vocab_mask, reduce=False, size_average=False)
        models[m].eval()
        if args.cuda:
            nll_loss[m] = nll_loss[m].cuda()
            cross_entropy_loss[m] = cross_entropy_loss[m].cuda()
    epoch = 0

    train_data = list(zip(train_srcs['A'], train_tgt['A'], lm_scores['A'], lm_scores['B']))
    cum_lossA = cum_lossB = 0
    att_loss = 0
    ce_lossA_log = 0
    ce_lossB_log = 0
    t = 0
    hist_valid_scores = {}
    hist_valid_scores['A'] = []
    hist_valid_scores['B'] = []

    patience = {}
    patience['A'] = patience['B'] = 0
    decay = {}
    decay['A'] = 0
    decay['B'] = 0
    while True:
        epoch += 1
        print('\nstart of epoch {:d}'.format(epoch))

        data = {}
        data['A'] = data_iter_for_dual(train_data, batch_size=args.batch_size, shuffle = False)
        
        for batchA in data['A']:
            src_sentsA, tgt_sentsA, src_scoresA, src_scoresB = batchA[0], batchA[1], batchA[2], batchA[3]
            tgt_sents_forA = [['<s>'] + sent + ['</s>'] for sent in tgt_sentsA]

            src_sents_varA, masksA = to_input_variable(src_sentsA, vocabs['A'].src, cuda=args.cuda)
            tgt_sents_varA, _ = to_input_variable(tgt_sents_forA, vocabs['A'].tgt, cuda=args.cuda)
            src_scores_varA = Variable(torch.FloatTensor(src_scoresA), requires_grad=False)
            

            src_sents_len_A = [len(s) for s in src_sentsA]
            # print(src_sents_varA, src_sents_len_A, tgt_sents_varA[:-1], masksA)
            scoresA, feature_A, att_sim_A = models['A'](src_sents_varA, src_sents_len_A, tgt_sents_varA[:-1], masksA)

            ce_lossA = cross_entropy_loss['A'](scoresA.view(-1, scoresA.size(2)), tgt_sents_varA[1:].view(-1)).cpu()
            
            batch_data = (src_sentsA, tgt_sentsA, src_scoresA, src_scoresB)
            src_sentsA, tgt_sentsA, src_scoresA, src_scoresB = get_new_batch(batch_data)
            tgt_sents_forB = [['<s>'] + sent + ['</s>'] for sent in src_sentsA]

            src_sents_varB, masksB = to_input_variable(tgt_sentsA, vocabs['B'].src, cuda=args.cuda)
            tgt_sents_varB, _ = to_input_variable(tgt_sents_forB, vocabs['B'].tgt, cuda=args.cuda)
            src_scores_varB = Variable(torch.FloatTensor(src_scoresB), requires_grad=False)
            
            src_sents_len = [len(s) for s in tgt_sentsA]
            scoresB, feature_B, att_sim_B = models['B'](src_sents_varB, src_sents_len, tgt_sents_varB[:-1], masksB)

            ce_lossB = cross_entropy_loss['B'](scoresB.view(-1, scoresB.size(2)), tgt_sents_varB[1:].view(-1)).cpu()

            optimizerA = optimizers['A']
            optimizerB = optimizers['B']

            optimizerA.zero_grad()
            optimizerB.zero_grad()
            # print (ce_lossA.size(), src_scores_varA.size(), tgt_sents_varA[1:].size(0))
            ce_lossA = ce_lossA.view(tgt_sents_varA[1:].size(0), tgt_sents_varA[1:].size(1)).mean(0)
            ce_lossB = ce_lossB.view(tgt_sents_varB[1:].size(0), tgt_sents_varB[1:].size(1)).mean(0)
            
            att_sim_A = torch.cat(att_sim_A, 1)

            masksA = masksA.transpose(1,0).unsqueeze(1)
            masksA = masksA.expand(masksA.size(0), att_sim_A.size(1), masksA.size(2))
            assert att_sim_A.size() == masksA.size(), '{} {}'.format(att_sim_A.size(), masksA.size())
            att_sim_B = torch.cat(att_sim_B, 1)
            masksB = masksB.transpose(1,0).unsqueeze(1)
            masksB = masksB.expand(masksB.size(0), att_sim_B.size(1), masksB.size(2))
            assert att_sim_B.size() == masksB.size(), '{} {}'.format(att_sim_B.size(), masksB.size())
            att_sim_B = att_sim_B.transpose(2, 1)
            loss_att_A = loss_att(att_sim_A, att_sim_B, masksB.transpose(1,0), src_sents_len)
            loss_att_B = loss_att(att_sim_A.transpose(2, 1), att_sim_B.transpose(2, 1), masksB, src_sents_len_A)

            dual_loss = (src_scores_varA - ce_lossA - src_scores_varB + ce_lossB) ** 2
            att_loss_ = (loss_att_A + loss_att_B)

            lossA = ce_lossA + args.beta1 * dual_loss + args.beta3 * att_loss_
            lossB = ce_lossB + args.beta2 * dual_loss + args.beta4 * att_loss_

            lossA = torch.mean(lossA)
            lossB = torch.mean(lossB)
            
            cum_lossA += lossA.data[0]
            cum_lossB += lossB.data[0]

            ce_lossA_log += torch.mean(ce_lossA).data[0]
            ce_lossB_log += torch.mean(ce_lossB).data[0]
            att_loss += (torch.mean(loss_att_A).data[0] + torch.mean(loss_att_B).data[0])

            optimizerA.zero_grad()
            lossA.backward(retain_graph=True)
            grad_normA = torch.nn.utils.clip_grad_norm(models['A'].parameters(), args.clip_grad)
            optimizerA.step()
            optimizerB.zero_grad()
            lossB.backward()
            grad_normB = torch.nn.utils.clip_grad_norm(models['B'].parameters(), args.clip_grad)
            optimizerB.step()
            if t % args.log_n_iter == 0 and t != 0:
                print('epoch %d, avg. loss A %.3f, avg. word loss A %.3f, avg, loss B %.3f, avg. word loss B %.3f, avg att loss %.3f' % (epoch, 
                        cum_lossA/args.log_n_iter , ce_lossA_log/args.log_n_iter, cum_lossB / args.log_n_iter, ce_lossB_log/args.log_n_iter,att_loss / args.log_n_iter))
                cum_lossA = 0
                cum_lossB = 0
                att_loss = 0
                ce_lossA_log = 0
                ce_lossB_log = 0
            if t % args.val_n_iter == 0 and t != 0:
                print ('Validation begins ...')
                for i, model_id in enumerate(['A', 'B']):
                    models[model_id].eval()

                    tmp_dev_data = dev_data[model_id]
                    dev_hyps = decode(models[model_id], tmp_dev_data)
                    dev_hyps = [hyps[0] for hyps in dev_hyps]
                    valid_metric = get_bleu([tgt for src, tgt in tmp_dev_data], dev_hyps, 'test')
                    models[model_id].train()
                    hist_scores = hist_valid_scores[model_id]
                    print ('Model_id {} Sentence bleu : {}'.format(model_id, valid_metric))
                    
                    is_better = len(hist_scores) == 0 or valid_metric > max(hist_scores)
                    hist_scores.append(valid_metric)

                    if not is_better:
                        patience[model_id] += 1
                        print('hit patience %d' % patience[model_id])
                        if patience[model_id] > 0:
                            if abs(optimizers[model_id].param_groups[0]['lr']) < 1e-8:
                                exit(0)
                            if decay[model_id] < 1:
                                lr = optimizers[model_id].param_groups[0]['lr'] * 0.5
                                print('Decay learning rate to %f' % lr)
                                optimizers[model_id].param_groups[0]['lr'] = lr
                                patience[model_id] = 0
                                decay[model_id] += 1
                            else:
                                for param in models[model_id].parameters():
                                    if param.size()[0] == 50000 or param.size()[0] == 27202:
                                        param.requires_grad = False

                                lr = optimizers[model_id].param_groups[0]['lr'] * 0.95
                                print('Decay learning rate to %f' % lr)
                                optimizers[model_id].param_groups[0]['lr'] = lr
                                decay[model_id] += 1
                            
                    else:
                        patience[model_id] = 0
                        if model_id == 'A':
                            np.save('{}.iter{}'.format(args.model[i], t), att_sim_A[0].cpu().data.numpy())
                        if model_id == 'B':
                            np.save('{}.iter{}'.format(args.model[i], t), att_sim_B[0].cpu().data.numpy())
                        models[model_id].save('{}.iter{}.bin'.format(args.model[i], t))

            t += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nmt', nargs=2, required=True, help='pre-train nmt model path')
    parser.add_argument('--src', nargs=2, required=True, help='training data path')
    parser.add_argument('--val', nargs=2, required=True, help='validation data path')
    parser.add_argument('--model', nargs=2, type=str, default=['modelA', 'modelB'])
    parser.add_argument('--log_n_iter', type=int, default=100)
    parser.add_argument('--val_n_iter', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=0.2)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', default=20, type=int, help='batch size')
    parser.add_argument('--beta1', default=1e-3, type=float, help='lambda1')
    parser.add_argument('--beta2', default=1e-2, type=float, help='lambda2')
    parser.add_argument('--beta3', default=1e-2, type=float, help='lambda3')
    parser.add_argument('--beta4', default=1e-1, type=float, help='lambda4')
    parser.add_argument('--clip_grad', default=5., type=float, help='clip gradients')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    print(args)

    dual(args)
