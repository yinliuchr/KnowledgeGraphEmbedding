from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import random

import numpy as np
import torch

from torch.utils.data import DataLoader

from model import KGEModel

from dataloader import TrainDataset
from dataloader import BidirectionalOneShotIterator

a_data_path = "../data/FB15k"

def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples


# construct 2 dicts: entity->id, relation->id
with open(os.path.join(a_data_path, 'entities.dict')) as fin:
    entity2id = dict()
    for line in fin:
        eid, entity = line.strip().split('\t')
        entity2id[entity] = int(eid)

with open(os.path.join(a_data_path, 'relations.dict')) as fin:
    relation2id = dict()
    for line in fin:
        rid, relation = line.strip().split('\t')
        relation2id[relation] = int(rid)


nentity = len(entity2id)
nrelation = len(relation2id)

a_nentity = nentity
a_nrelation = nrelation

train_triples = read_triple(os.path.join(a_data_path, 'train.txt'), entity2id, relation2id)
valid_triples = read_triple(os.path.join(a_data_path, 'valid.txt'), entity2id, relation2id)
test_triples = read_triple(os.path.join(a_data_path, 'test.txt'), entity2id, relation2id)

all_true_triples = train_triples + valid_triples + test_triples

a_model = "RotatE"
a_hidden_dim = 1000
a_gamma = 24.0

a_batchsize = 1024
a_neg_sample_size = 256
a_alpha = 1.0
a_lr = 0.0001
a_max_steps = 150000
a_test_batchsize = 16

kge_model = KGEModel(
    model_name=a_model,
    nentity=nentity,
    nrelation=nrelation,
    hidden_dim=a_hidden_dim,
    gamma=a_gamma,
    double_entity_embedding=True,
    double_relation_embedding=False
)

train_dataloader_head = DataLoader(
    TrainDataset(train_triples, nentity, nrelation, a_neg_sample_size, 'head-batch'),
    batch_size=a_batchsize,
    shuffle=True,
    # num_workers=4,
    collate_fn=TrainDataset.collate_fn
)

train_dataloader_tail = DataLoader(
    TrainDataset(train_triples, nentity, nrelation, a_neg_sample_size, 'tail-batch'),
    batch_size=a_batchsize,
    shuffle=True,
    # num_workers=4,
    collate_fn=TrainDataset.collate_fn
)

train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)

#############################################
for i in range(4):
    positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)
    print(positive_sample.size())
    print(positive_sample[0])
    print(positive_sample[1023])
    print(negative_sample.size())
    print(negative_sample[0])
    print(subsampling_weight.size())
    print(mode, '\n\n')

# current_learning_rate = a_lr
# optimizer = torch.optim.Adam(
#     filter(lambda p: p.requires_grad, kge_model.parameters()),
#     lr=current_learning_rate
# )
#
# warm_up_steps = a_max_steps // 2
#
# init_step = 0
# step = 0
#
#
# # # Training Loop
# # for step in range(init_step, a_max_steps):
# #
# #     log = kge_model.train_step(kge_model, optimizer, train_iterator, args=None)
# #
# #     if step >= warm_up_steps:
# #         current_learning_rate = current_learning_rate / 10
# #         logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
# #         optimizer = torch.optim.Adam(
# #             filter(lambda p: p.requires_grad, kge_model.parameters()),
# #             lr=current_learning_rate
# #         )
# #         warm_up_steps = warm_up_steps * 3
#
#
# def train_step(model, optimizer, train_iterator, args):
#     '''
#     A single train step. Apply back-propation and return the loss
#     '''
#
#     model.train()
#
#     optimizer.zero_grad()
#
#     positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)
#
#     if args.cuda:
#         positive_sample = positive_sample.cuda()
#         negative_sample = negative_sample.cuda()
#         subsampling_weight = subsampling_weight.cuda()
#
#     negative_score = model((positive_sample, negative_sample), mode=mode)
#
#     if args.negative_adversarial_sampling:
#         # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
#         negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
#                           * F.logsigmoid(-negative_score)).sum(dim=1)
#     else:
#         negative_score = F.logsigmoid(-negative_score).mean(dim=1)
#
#     positive_score = model(positive_sample)
#
#     positive_score = F.logsigmoid(positive_score).squeeze(dim=1)
#
#     if args.uni_weight:
#         positive_sample_loss = - positive_score.mean()
#         negative_sample_loss = - negative_score.mean()
#     else:
#         positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
#         negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()
#
#     loss = (positive_sample_loss + negative_sample_loss) / 2
#
#     if args.regularization != 0.0:
#         # Use L3 regularization for ComplEx and DistMult
#         regularization = args.regularization * (
#                 model.entity_embedding.norm(p=3) ** 3 +
#                 model.relation_embedding.norm(p=3).norm(p=3) ** 3
#         )
#         loss = loss + regularization
#         regularization_log = {'regularization': regularization.item()}
#     else:
#         regularization_log = {}
#
#     loss.backward()
#
#     optimizer.step()
#
#     log = {
#         **regularization_log,
#         'positive_sample_loss': positive_sample_loss.item(),
#         'negative_sample_loss': negative_sample_loss.item(),
#         'loss': loss.item()
#     }
#
#     return log
#
#
# @staticmethod
# def test_step(model, test_triples, all_true_triples, args):
#     '''
#     Evaluate the model on test or valid datasets
#     '''
#
#     model.eval()
#
#     if args.countries:
#         # Countries S* datasets are evaluated on AUC-PR
#         # Process test data for AUC-PR evaluation
#         sample = list()
#         y_true = list()
#         for head, relation, tail in test_triples:
#             for candidate_region in args.regions:
#                 y_true.append(1 if candidate_region == tail else 0)
#                 sample.append((head, relation, candidate_region))
#
#         sample = torch.LongTensor(sample)
#         if args.cuda:
#             sample = sample.cuda()
#
#         with torch.no_grad():
#             y_score = model(sample).squeeze(1).cpu().numpy()
#
#         y_true = np.array(y_true)
#
#         # average_precision_score is the same as auc_pr
#         auc_pr = average_precision_score(y_true, y_score)
#
#         metrics = {'auc_pr': auc_pr}
#
#     else:
#         # Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
#         # Prepare dataloader for evaluation
#         test_dataloader_head = DataLoader(
#             TestDataset(
#                 test_triples,
#                 all_true_triples,
#                 args.nentity,
#                 args.nrelation,
#                 'head-batch'
#             ),
#             batch_size=args.test_batch_size,
#             num_workers=max(1, args.cpu_num // 2),
#             collate_fn=TestDataset.collate_fn
#         )
#
#         test_dataloader_tail = DataLoader(
#             TestDataset(
#                 test_triples,
#                 all_true_triples,
#                 args.nentity,
#                 args.nrelation,
#                 'tail-batch'
#             ),
#             batch_size=args.test_batch_size,
#             num_workers=max(1, args.cpu_num // 2),
#             collate_fn=TestDataset.collate_fn
#         )
#
#         test_dataset_list = [test_dataloader_head, test_dataloader_tail]
#
#         logs = []
#
#         step = 0
#         total_steps = sum([len(dataset) for dataset in test_dataset_list])
#
#         with torch.no_grad():
#             for test_dataset in test_dataset_list:
#                 for positive_sample, negative_sample, filter_bias, mode in test_dataset:
#                     if args.cuda:
#                         positive_sample = positive_sample.cuda()
#                         negative_sample = negative_sample.cuda()
#                         filter_bias = filter_bias.cuda()
#
#                     batch_size = positive_sample.size(0)
#
#                     score = model((positive_sample, negative_sample), mode)
#                     score += filter_bias
#
#                     # Explicitly sort all the entities to ensure that there is no test exposure bias
#                     argsort = torch.argsort(score, dim=1, descending=True)
#
#                     if mode == 'head-batch':
#                         positive_arg = positive_sample[:, 0]
#                     elif mode == 'tail-batch':
#                         positive_arg = positive_sample[:, 2]
#                     else:
#                         raise ValueError('mode %s not supported' % mode)
#
#                     for i in range(batch_size):
#                         # Notice that argsort is not ranking
#                         ranking = (argsort[i, :] == positive_arg[i]).nonzero()
#                         assert ranking.size(0) == 1
#
#                         # ranking + 1 is the true ranking used in evaluation metrics
#                         ranking = 1 + ranking.item()
#                         logs.append({
#                             'MRR': 1.0 / ranking,
#                             'MR': float(ranking),
#                             'HITS@1': 1.0 if ranking <= 1 else 0.0,
#                             'HITS@3': 1.0 if ranking <= 3 else 0.0,
#                             'HITS@10': 1.0 if ranking <= 10 else 0.0,
#                         })
#
#                     if step % args.test_log_steps == 0:
#                         logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))
#
#                     step += 1
#
#         metrics = {}
#         for metric in logs[0].keys():
#             metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
#
#     return metrics
