
#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.init import xavier_normal_, xavier_uniform_

from sklearn.metrics import average_precision_score

from torch.utils.data import DataLoader

from dataloader import TestDataset

class ConvModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma,
                 double_entity_embedding=False, double_relation_embedding=False):
        super(ConvModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation

        self.entity_dim = hidden_dim
        self.relation_dim = hidden_dim
        self.embedding_dim = hidden_dim

        # self.entity_embedding_real = nn.Embedding(self.nentity, self.entity_dim, padding_idx=0 )
        # self.entity_embedding_img = nn.Embedding(self.nentity, self.entity_dim, padding_idx=0 )
        self.entity_embedding = nn.Embedding(self.nentity, self.entity_dim, padding_idx=0 )

        # nn.init.uniform_(
        #     tensor=self.entity_embedding,
        #     a=-self.embedding_range.item(),
        #     b=self.embedding_range.item()
        # )

        # self.relation_embedding_real = nn.Embedding(self.nrelation, self.relation_dim, padding_idx=0 )
        # self.relation_embedding_img = nn.Embedding(self.nrelation, self.relation_dim, padding_idx=0 )
        self.relation_embedding = nn.Embedding(self.nrelation, self.relation_dim, padding_idx=0 )

        self.inp_drop = torch.nn.Dropout(.2)
        self.hidden_drop = torch.nn.Dropout(.3)
        self.feature_map_drop = torch.nn.Dropout2d(.2)
        self.loss = torch.nn.BCELoss()  # modify: cosine embedding loss / triplet loss
        self.emb_dim1 = 20              # this is from the original configuration in ConvE
        self.emb_dim2 = self.embedding_dim // self.emb_dim1

        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=True)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(self.embedding_dim)
        self.register_parameter('b', nn.Parameter(torch.zeros(self.nentity)))
        self.fc = torch.nn.Linear(14848, self.embedding_dim)

    def init(self):
        xavier_normal_(self.entity_embedding.weight.data)
        xavier_normal_(self.relation_embedding.weight.data)


    def forward(self, sample):              # sample is  size: (... ,3)

        # batch_size = sample.size(0)
        head = sample[:, 0]
        relation = sample[:,1]
        tail = sample[:,2]

        e1_embedded= self.entity_embedding(head).view(-1, 1, self.emb_dim1, self.emb_dim2)
        rel_embedded = self.relation_embedding(relation).view(-1, 1, self.emb_dim1, self.emb_dim2)
        e2_embedded = self.entity_embedding(tail).view(-1, 1, self.emb_dim1, self.emb_dim2)

        print('\n**************************\ne1_dim: ', e1_embedded.size(),
              '\trel: ', rel_embedded.size(),
              '\te2: ', e2_embedded.size())

        stacked_inputs = torch.cat([e1_embedded, rel_embedded, e2_embedded], 2)   # shape: [size, 1, 20*3, 10]
        print('init_x_size: ', stacked_inputs.size())

        stacked_inputs = self.bn0(stacked_inputs)
        print(stacked_inputs.size())
        x= self.inp_drop(stacked_inputs)
        print(x.size())
        x= self.conv1(x)
        print(x.size())
        x= self.bn1(x)
        print(x.size())
        x= F.relu(x)
        print(x.size())
        x = self.feature_map_drop(x)
        print(x.size())
        x = x.view(x.shape[0], -1)
        print(x.size())
        x = self.fc(x)
        print(x.size())
        x = self.hidden_drop(x)
        print(x.size())
        x = self.bn2(x)
        print(x.size())
        x = F.relu(x)
        print(x.size())
        x = torch.mm(x, self.entity_embedding.weight.transpose(1,0))
        print(x.size())
        x += self.b.expand_as(x)
        print(x.size())
        pred = torch.sigmoid(x)
        print(x.size())

        print('************************* \n')

        return pred



    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)
        # mode = 'single'
        bs = positive_sample.size(0)        # e.g., 1024
        ns = negative_sample.size(1)        # e.g., 256

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        positive_score = model(positive_sample)
        # positive_score = 1.0 - positive_score
        # positive_score = F.logsigmoid(positive_score).squeeze(dim=1)


        neg_sam = []
        if mode == 'head-batch':
            for i in range(bs):
                ori_sam = positive_sample[i].repeat(ns, 1)      # 256 * 3
                ori_sam[:,0] = negative_sample[i]
                neg_sam.append(ori_sam)
        elif mode == 'tail-batch':
            for i in range(bs):
                ori_sam = positive_sample[i].repeat(ns,1)
                ori_sam[:,2] = negative_sample[i]
                neg_sam.append(ori_sam)

        neg_sam = torch.cat(neg_sam, dim=0)
        print(neg_sam.size())

        if args.cuda:
            neg_sam = neg_sam.cuda()
        negative_score = model(neg_sam)

        positive_sample_loss = 1.0 - positive_score.mean()
        negative_sample_loss = negative_score.mean()
        loss = positive_sample_loss + negative_sample_loss

        # negative_score = model((positive_sample, negative_sample), mode=mode)
        #
        # if args.negative_adversarial_sampling:
        #     # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
        #     negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
        #                       * F.logsigmoid(-negative_score)).sum(dim=1)
        # else:
        #     negative_score = F.logsigmoid(-negative_score).mean(dim=1)



        # if args.uni_weight:
        #     positive_sample_loss = - positive_score.mean()
        #     negative_sample_loss = - negative_score.mean()
        # else:
        #     positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
        #     negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()
        #
        # loss = (positive_sample_loss + negative_sample_loss) / 2

        if args.regularization != 0.0:
            # Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                    model.entity_embedding.norm(p=3) ** 3 +
                    model.relation_embedding.norm(p=3).norm(p=3) ** 3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}

        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log





    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):
        '''
        Evaluate the model on test or valid datasets
        '''

        model.eval()

        if args.countries:
            # Countries S* datasets are evaluated on AUC-PR
            # Process test data for AUC-PR evaluation
            sample = list()
            y_true = list()
            for head, relation, tail in test_triples:
                for candidate_region in args.regions:
                    y_true.append(1 if candidate_region == tail else 0)
                    sample.append((head, relation, candidate_region))

            sample = torch.LongTensor(sample)
            if args.cuda:
                sample = sample.cuda()

            with torch.no_grad():
                y_score = model(sample).squeeze(1).cpu().numpy()

            y_true = np.array(y_true)

            # average_precision_score is the same as auc_pr
            auc_pr = average_precision_score(y_true, y_score)

            metrics = {'auc_pr': auc_pr}

        else:
            # Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
            # Prepare dataloader for evaluation
            test_dataloader_head = DataLoader(
                TestDataset(
                    test_triples,
                    all_true_triples,
                    args.nentity,
                    args.nrelation,
                    'head-batch'
                ),
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num // 2),
                collate_fn=TestDataset.collate_fn
            )

            test_dataloader_tail = DataLoader(
                TestDataset(
                    test_triples,
                    all_true_triples,
                    args.nentity,
                    args.nrelation,
                    'tail-batch'
                ),
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num // 2),
                collate_fn=TestDataset.collate_fn
            )

            test_dataset_list = [test_dataloader_head, test_dataloader_tail]

            logs = []

            step = 0
            total_steps = sum([len(dataset) for dataset in test_dataset_list])

            with torch.no_grad():
                for test_dataset in test_dataset_list:
                    for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                        if args.cuda:
                            positive_sample = positive_sample.cuda()
                            negative_sample = negative_sample.cuda()
                            filter_bias = filter_bias.cuda()

                        batch_size = positive_sample.size(0)

                        # positive_score = model(positive_sample)

                        bs, ns = negative_sample.size(0), negative_sample.size(1)       # 16, 14951

                        ori_sam = None
                        score = []
                        if mode == 'head-batch':
                            for i in range(bs):
                                ori_sam = positive_sample[i].repeat(ns, 1)  # 14541 * 3
                                ori_sam[:, 0] = negative_sample[i]
                                if args.cuda:
                                    ori_sam = ori_sam.cuda()
                                print('\n**************************\nori_sam_dim: ', ori_sam.size(),
                                      'Model(ori_sam)_dim: ', model(ori_sam).size(),
                                      '\t\tfilterbias[i].size: ', filter_bias[i].size(),
                                      '\n**************************\n')
                                temp_score = model(ori_sam) + filter_bias[i]        # size: (14951)
                                print('\n**************************\nTemp_score_dim: ', temp_score.size(),
                                      '\n**************************\n')
                                score.append(temp_score)

                                # neg_sam.append(ori_sam)
                        elif mode == 'tail-batch':
                            for i in range(bs):
                                ori_sam = positive_sample[i].repeat(ns, 1)  # 14951 * 3
                                ori_sam[:, 2] = negative_sample[i]
                                if args.cuda:
                                    ori_sam = ori_sam.cuda()
                                print('\n**************************\nori_sam_dim: ', ori_sam.size(),
                                      'Model(ori_sam)_dim: ', model(ori_sam).size(),
                                      '\t\tfilterbias[i].size: ', filter_bias[i].size(),
                                      '\n**************************\n')
                                temp_score = model(ori_sam) + filter_bias[i]
                                print('\n**************************\nTemp_score_dim: ', temp_score.size(),
                                      '\n**************************\n')
                                score.append(temp_score)
                                # neg_sam.append(ori_sam)

                        score = torch.cat(score, dim=0)             # 16 * 14951
                        # neg_sam = torch.cat(neg_sam, dim=0)         # (16 * 14951) * 3

                        # score = model((positive_sample, negative_sample), mode)
                        # score += filter_bias

                        # Explicitly sort all the entities to ensure that there is no test exposure bias
                        print('\n**************************\nScore_dim: ', score.size(), '\n**************************\n')

                        argsort = torch.argsort(score, dim=1, descending=True)

                        if mode == 'head-batch':
                            positive_arg = positive_sample[:, 0]
                        elif mode == 'tail-batch':
                            positive_arg = positive_sample[:, 2]
                        else:
                            raise ValueError('mode %s not supported' % mode)

                        for i in range(batch_size):
                            # Notice that argsort is not ranking
                            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                            assert ranking.size(0) == 1

                            # ranking + 1 is the true ranking used in evaluation metrics
                            ranking = 1 + ranking.item()
                            logs.append({
                                'MRR': 1.0 / ranking,
                                'MR': float(ranking),
                                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                                'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            })

                        if step % args.test_log_steps == 0:
                            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                        step += 1

            metrics = {}
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs]) / len(logs)

        return metrics
