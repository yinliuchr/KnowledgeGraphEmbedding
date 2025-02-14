#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score

from torch.utils.data import DataLoader

from dataloader import TestDataset

# bash run.sh train RotatE FB15k    0       0      1024        256               1000         24.0    1.0   0.0001 150000         16               -de
#               1     2      3       4      5        6          7                   8          9       10     11     12           13
#              mode model  dataset  GPU  saveid    batchsize   neg_sample_size  hidden_dim    gamma   alpha   lr    Max_steps  test_batchsize

class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma, 
                 double_entity_embedding=False, double_relation_embedding=False):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )
        
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), 
            requires_grad=False
        )
        
        self.entity_dim = hidden_dim*2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim*2 if double_relation_embedding else hidden_dim

        if model_name == 'QuarterNion':
            self.entity_dim = hidden_dim * 4
            self.relation_dim = hidden_dim * 4

        if model_name == 'QuarterRotatE':
            self.entity_dim = hidden_dim * 3
            self.relation_dim = hidden_dim * 4
        
        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        
        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )

        ##############################################################################################################
        if model_name == 'DistMultC':
            self.fc = nn.Linear(self.hidden_dim, 1)
            # self.register_parameter('u', nn.Parameter(torch.zeros(256)))
        ##############################################################################################################

        if model_name == 'ComplExD':
            self.fc = nn.Linear(4,1)

        if model_name == 'ComplExG':
            self.fc = nn.Linear(8,1)

        if model_name == 'pRotatE':
            self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))
        
        #Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'DistMultC', 'ComplEx', 'QuarterNion', 'ComplExC', 'ComplExD','ComplExH', 'RotatE', 'QuarterRotatE', 'pRotatE']:
            raise ValueError('model %s not supported' % model_name)
            
        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name in {'ComplEx', 'QuarterNion', 'ComplExC', 'ComplExD', 'ComplExH'} and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError(model_name + ' should use --double_entity_embedding and --double_relation_embedding')
        
    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements 
        in their triple ((head, relation) or (relation, tail)).
        '''

        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=sample[:,1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,2]
            ).unsqueeze(1)
            
        elif mode == 'head-batch':
            tail_part, head_part = sample           # tail part: 1024 * 3 (1024 positive triples)
                                                    # head part: 1024 * 256 (each row represent neg sample ids of the corresponding positive triple)
                                                    # in other words, each positive triplet have 256 negetive triplets
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)     # 1024 256
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)                # indexes * entity_dim: (1024 * 256) * entity_dim
                                                                        # corrupted head
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=tail_part[:, 1]
            ).unsqueeze(1)                                              # 1024 * 1 * entity_dim
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part[:, 2]
            ).unsqueeze(1)                                              # 1024 * 1 * entity_dim
            
        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part[:, 0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            
        else:
            raise ValueError('mode %s not supported' % mode)
            
        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'DistMultC': self.DistMultC,
            'ComplEx': self.ComplEx,
            'QuarterNion': self.QuarterNion,
            'ComplExC': self.ComplExC,
            'ComplExD': self.ComplExD,
            'ComplExH': self.ComplExH,
            'RotatE': self.RotatE,
            'QuarterRotatE': self.QuarterRotatE,
            'pRotatE': self.pRotatE
        }
        
        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)
        
        return score
    
    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim = 2)
        return score

    def DistMultC(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)        # bs * 256 * dim
        else:
            score = (head * relation) * tail        # bs * 256 * dim

        score = self.fc(score).squeeze(dim=2)       # bs * 256

        # score += self.u.expand_as(score)

        return score


    def ComplEx(self, head, relation, tail, mode):          # rrr + rii + iri - iir
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':        #  re/im_relation, re/im_tail: bs * 1 * dim,  re/im_head: bs * 256 * dim
            re_score = re_relation * re_tail + im_relation * im_tail        # re_score: bs * 1 * dim
            im_score = re_relation * im_tail - im_relation * re_tail        # im_score: bs * 1 * dim
            score = re_head * re_score + im_head * im_score                 # re/im_score: bs * 1 * dim, re/im_head: bs * 256 * dim, => score: bs * 256 * dim
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim = 2)          # score = bs * 256,  ( bs * 256 * dim => bs * 256)
        return score


    def QuarterNion(self, head, relation, tail, mode):      # Re < h, r, t* >
        h1, h2, h3, h4 = torch.chunk(head, 4, dim=2)
        r1, r2, r3, r4 = torch.chunk(relation, 4, dim=2)
        t1, t2, t3, t4 = torch.chunk(tail, 4, dim=2)
        if mode == 'head-batch':  # first compute  s = < r, t* > =  < (r1 + r2 i +  r3 j + r4 k), ( t1 - t2 i - t3 j - t4 k) >
            s1 = r1 * t1 + r2 * t2 + r3 * t3 + r4 * t4          # real part
            s2 = - r1 * t2 + r2 * t1 - r3 * t4 + r4 * t3        # i part
            s3 = - r1 * t3 + r3 * t1 + r2 * t4 - r4 * t2        # j part
            s4 = - r1 * t4 + r4 * t1 - r2 * t3 + r3 * t2        # k part
            # now compute RE < (h1 + h2 i + h3 j + h4 k), (s1 + s2 i + s3 j + s4 k) >
            score = h1 * s1 - h2 * s2 - h3 * s3 - h4 * s4
        else:       # first compute <h, r> = < (h1 + h2 i + h3 j + h4 k), (r1 + r2 i + r3 j + r4 k) >
            s1 = h1 * r1 - h2 * r2 - h3 * r3 - h4 * r4
            s2 = h1 * r2 + h2 * r1 + h3 * r4 - h4 * r3
            s3 = h1 * r3 + h3 * r1 - h2 * r4 + h4 * r2
            s4 = h1 * r4 + h4 * r1 + h2 * r3 - h3 * r2
            # now compute Re <s, t*> = Re < (s1 + s2 i + s3 j + s4 k), (t1 - t2 i - t3 j - t4 k) >
            score = s1 * t1 + s2 * t2 + s3 * t3 + s4 * t4

        score = score.sum(dim=2)
        return score


    def ComplExC(self, head, relation, tail, mode):             # rrr - rii - iri - iir
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':        #  re/im_relation, re/im_tail: bs * 1 * dim,  re/im_head: bs * 256 * dim
            re_score = re_relation * re_tail - im_relation * im_tail        # re_score: bs * 1 * dim
            im_score = re_relation * im_tail + im_relation * re_tail        # im_score: bs * 1 * dim
            score = re_head * re_score - im_head * im_score                 # re/im_score: bs * 1 * dim, re/im_head: bs * 256 * dim, => score: bs * 256 * dim
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail - im_score * im_tail

        score = score.sum(dim = 2)          # score = bs * 256
        return score


    def ComplExD(self, head, relation, tail, mode):          # rrr + rii + iri - iir
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode =='head-batch':
            rrr = re_head * (re_relation * re_tail)       # bs * 256 * dim
            rii = re_head * (im_relation * im_tail)
            iri = im_head * (re_relation * im_tail)
            iir = im_head * (im_relation * re_tail)
        else:
            rrr = (re_head * re_relation) * re_tail  # bs * 256 * dim
            rii = (re_head * im_relation) * im_tail
            iri = (im_head * re_relation) * im_tail
            iir = (im_head * im_relation) * re_tail

        res = torch.stack((rrr,rii, iri, iir), 3)   # bs * 256 * dim * 4

        res = self.fc(res).squeeze(dim=3)            # bs * 256 * dim

        # print('\n\n ######## \n\n res: ', res.shape)

        score = res.sum(dim = 2)          # score = bs * 256
        return score

    def ComplExH(self, head, relation, tail, mode):
        h1, h2 = torch.chunk(head, 2, dim=2)
        r1, r2 = torch.chunk(relation, 2, dim=2)
        t1, t2 = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            a111 = h1 * (r1 * t1)
            a112 = h1 * (r1 * t2)
            a121 = h1 * (r2 * t1)
            a122 = h1 * (r2 * t2)
            a211 = h2 * (r1 * t1)
            a212 = h2 * (r1 * t2)
            a221 = h2 * (r2 * t1)
            a222 = h2 * (r2 * t2)
        else:
            a111 = h1 * r1 * t1
            a112 = h1 * r1 * t2
            a121 = h1 * r2 * t1
            a122 = h1 * r2 * t2
            a211 = h2 * r1 * t1
            a212 = h2 * r1 * t2
            a221 = h2 * r2 * t1
            a222 = h2 * r2 * t2
        score = a111 + a112 - a121 + a122 - a211 - a212 + a221 - a222
        score = score.sum(dim=2)
        return score

    # def ComplExG(self, head, relation, tail, mode):
    #     h1, h2 = torch.chunk(head, 2, dim=2)
    #     r1, r2 = torch.chunk(relation, 2, dim=2)
    #     t1, t2 = torch.chunk(tail, 2, dim=2)
    #
    #     a111 = h1 * r1 * t1
    #     a112 = h1 * r1 * t2
    #     a121 = h1 * r2 * t1
    #     a122 = h1 * r2 * t2
    #     a211 = h2 * r1 * t1
    #     a212 = h2 * r1 * t2
    #     a221 = h2 * r2 * t1
    #     a222 = h2 * r2 * t2
    #
    #
    #     res = torch.stack((a111, a112, a121, a122, a211, a212, a221, a222), 3)  # bs * 256 * dim * 4
    #
    #     res = self.fc(res).squeeze(dim=3)  # bs * 256 * dim
    #
    #     # print('\n\n ######## \n\n res: ', res.shape)
    #
    #     score = res.sum(dim=2)  # score = bs * 256
    #     return score

    def RotatE(self, head, relation, tail, mode):
        # head (if corrupted): 1024 * 256 * ent_dim         (ent_dim = hidden_dim * 2)
        # relation: 1024 * 1 * hidden_dim
        # tail: 1024 * 1 * ent_dim
        # mode: (here assume to be 'head_batch')

        pi = 3.14159265358979323846
        
        re_head, im_head = torch.chunk(head, 2, dim=2)          # both 1024 * 256 * hid_dim
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)          # both 1024 * 1 * hid_dim

        #Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)             # 1024 * 1 * hid_dim
        im_relation = torch.sin(phase_relation)             # 1024 * 1 * hid_dim

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail       # 1024 * 1 * hid_dim
            im_score = re_relation * im_tail - im_relation * re_tail        # 1024 * 1 * hid_dim
            re_score = re_score - re_head                                   # 1024 * 256 * hid_dim
            im_score = im_score - im_head                                   # 1024 * 256 * hid_dim
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)          # # 2 * 1024 * 256 * hid_dim
        score = score.norm(dim = 0)                                 # 1024 * 256 * hid_dim

        score = self.gamma.item() - score.sum(dim = 2)              # 1024 * 256
        return score

    def QuarterRotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846
        h1, h2, h3 = torch.chunk(head, 3, dim=2)        # head
        t1, t2, t3 = torch.chunk(tail, 3, dim=2)        # tail

        q0, q1, q2, q3 = torch.chunk(relation, 4, dim=2)    # relation, seen as a quarternion
        qn = torch.sqrt(q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2)
        q0, q1, q2, q3 = q0 / qn, q1 / qn, q2 / qn, q3 / qn     # q = q0 + (q1 i + q2 j + q3 k) is a unit quarternion

        if mode == 'head-batch':
            q1, q2, q3 = -q1, -q2, -q3          # this is important as tail should rotate back
            coeff1 = q0 ** 2 - (q1**2 + q2**2 + q3**2)
            coeff2 = 2 * (q1 * t1 + q2 * t2 + q3 * t3)
            coeff3 = 2 * q0
            # rot_tail = coeff1 * vec(t) + coeff2 * vec(q) + coeff3 * (vec(q) x vec(t))
            rot_tail_1 = coeff1 * t1 + coeff2 * q1 + coeff3 * (q2 * t3 - q3 * t2)
            rot_tail_2 = coeff1 * t2 + coeff2 * q2 + coeff3 * (q3 * t1 - q1 * t3)
            rot_tail_3 = coeff1 * t3 + coeff2 * q3 + coeff3 * (q1 * t2 - q2 * t1)

            dif1 = h1 - rot_tail_1
            dif2 = h2 - rot_tail_2
            dif3 = h3 - rot_tail_3

        else:
            coeff1 = q0 ** 2 - (q1 ** 2 + q2 ** 2 + q3 ** 2)
            coeff2 = 2 * (q1 * h1 + q2 * h2 + q3 * h3)
            coeff3 = 2 * q0
            # rot_head = coeff1 * vec(h) + coeff2 * vec(q) + coeff3 * (vec(q) x vec(h))
            rot_head_1 = coeff1 * h1 + coeff2 * q1 + coeff3 * (q2 * h3 - q3 * h2)
            rot_head_2 = coeff1 * h2 + coeff2 * q2 + coeff3 * (q3 * h1 - q1 * h3)
            rot_head_3 = coeff1 * h3 + coeff2 * q3 + coeff3 * (q1 * h2 - q2 * h1)

            dif1 = t1 - rot_head_1
            dif2 = t2 - rot_head_2
            dif3 = t3 - rot_head_3

        score = torch.stack([dif1, dif2, dif3], dim=0)
        score = score.norm(dim=0)
        score = self.gamma.item() - score.sum(dim=2)
        return score




    def pRotatE(self, head, relation, tail, mode):
        pi = 3.14159262358979323846
        
        #Make phases of entities and relations uniformly distributed in [-pi, pi]

        phase_head = head/(self.embedding_range.item()/pi)
        phase_relation = relation/(self.embedding_range.item()/pi)
        phase_tail = tail/(self.embedding_range.item()/pi)

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)            
        score = torch.abs(score)

        score = self.gamma.item() - score.sum(dim = 2) * self.modulus
        return score
    
    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        negative_score = model((positive_sample, negative_sample), mode=mode)

        if args.negative_adversarial_sampling:
            #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach() 
                              * F.logsigmoid(-negative_score)).sum(dim = 1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim = 1)

        positive_score = model(positive_sample)

        positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)

        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss)/2
        
        if args.regularization != 0.0:
            #Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                model.entity_embedding.norm(p = 3)**3 + 
                model.relation_embedding.norm(p = 3).norm(p = 3)**3
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
            #Countries S* datasets are evaluated on AUC-PR
            #Process test data for AUC-PR evaluation
            sample = list()
            y_true  = list()
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

            #average_precision_score is the same as auc_pr
            auc_pr = average_precision_score(y_true, y_score)

            metrics = {'auc_pr': auc_pr}
            
        else:
            #Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
            #Prepare dataloader for evaluation
            test_dataloader_head = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation, 
                    'head-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
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
                num_workers=max(1, args.cpu_num//2), 
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

                        score = model((positive_sample, negative_sample), mode)
                        score += filter_bias

                        # print('\n**************************\nScore_dim: ', score.size(), '\n**************************\n')


                        #Explicitly sort all the entities to ensure that there is no test exposure bias
                        argsort = torch.argsort(score, dim = 1, descending=True)

                        if mode == 'head-batch':
                            positive_arg = positive_sample[:, 0]
                        elif mode == 'tail-batch':
                            positive_arg = positive_sample[:, 2]
                        else:
                            raise ValueError('mode %s not supported' % mode)

                        for i in range(batch_size):
                            #Notice that argsort is not ranking
                            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                            assert ranking.size(0) == 1

                            #ranking + 1 is the true ranking used in evaluation metrics
                            ranking = 1 + ranking.item()
                            logs.append({
                                'MRR': 1.0/ranking,
                                'MR': float(ranking),
                                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                                'HITS@10': 1.0 if ranking <= 10 else 0.0,
                                'HITS@1000': 1.0 if ranking <= 1000 else 0.0
                            })

                        if step % args.test_log_steps == 0:
                            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                        step += 1

            metrics = {}
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs])/len(logs)

        return metrics
