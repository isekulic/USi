import argparse
import ast
import json
import numpy as np
import pandas as pd
import pickle
import os
import torch

from collections import defaultdict
from IPython import embed
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class ClariQDatasetDH(Dataset):
    def __init__(self, tokenizer, args, mode='dev'):
        self.tokenizer = tokenizer
        self.data_dir = args.data_dir
        self.mode = mode
        self.max_seq_len = args.max_seq_len
        self.hparams = args

        if args.test_mode and mode == 'test':
            mode = 'dev' #tmp; load dev data cuz it has facets
        if self.hparams.use_faceted_data:
            self.df = pd.read_csv(args.my_faceted_data +'.'+ mode, sep='\t')
        elif self.hparams.just_qulac:
            self.qulac = pd.read_json(self.hparams.qulac_json)
            self.data = json.load(open('../../qulac/data/qulac/qulac_hist012_dict.json'))

            # undersampling
            np.random.seed(42)
            sampled_keys = set(np.random.choice(list(self.data.keys()), 6000))
            self.data = {k: v for k, v in self.data.items() if k in sampled_keys}
            print('undersampled:', len(self.data))

            # split into train-dev-test
            all_topics = sorted(self.qulac.topic_id.unique())
            topics = dict()
            topics['train'] = set(np.random.choice(all_topics, 100, replace=False))
            tmp = set(all_topics) - topics['train']
            topics['dev'] = set(np.random.choice(sorted(tmp), 25, replace=False))
            # topics['test'] = tmp - set(topics['dev'])
            # temporary 
            topics['test'] = set(np.random.choice(list(topics['dev']), 5))

            self.data = {k: v for k, v in self.data.items() if int(k.split('-')[0]) in topics[mode]}
            print('{mode} filtered:', len(self.data))
            self.df = self.qulac[self.qulac['topic_id'].isin(topics[mode])]
            self.df = self.df.rename(columns={'topic': 'initial_request'})

        elif self.hparams.history_data and not self.hparams.human_data:
            self.qulac = pd.read_json(self.hparams.qulac_json)
            self.data = json.load(open(self.hparams.history_data, 'r'))
            print(mode, len(self.data))

            # undersampling
            np.random.seed(42)
            sampled_keys = set(np.random.choice(list(self.data.keys()), 10000))
            self.data = {k: v for k, v in self.data.items() if k in sampled_keys}
            print('undersampled:', len(self.data))

            # split into train-dev-test
            all_topics = sorted(self.qulac.topic_id.unique())
            topics = dict()
            topics['train'] = set(np.random.choice(all_topics, 100, replace=False))
            tmp = set(all_topics) - topics['train']
            topics['dev'] = set(np.random.choice(sorted(tmp), 25, replace=False))
            # topics['test'] = tmp - set(topics['dev'])
            # temporary 
            topics['test'] = set(np.random.choice(list(topics['dev']), 5))

            self.data = {k: v for k, v in self.data.items() if int(k.split('-')[0]) in topics[mode]}
            print('{mode} filtered:', len(self.data))
            self.qulac = self.qulac[self.qulac['topic_id'].isin(topics[mode])]
            # self.qulac.to_csv(f'{mode}_qulac.csv', index=False)
            self.idx_to_key = dict(enumerate(self.data.keys()))

            # for key in self.data:
                # sample = self.data[key] 
                # q = sample['query']
                # cq = sample['question']
                # ans = sample['answer']
                # hist = sample['history_list']
                # # extract facets by key id
                # topic_facet_id = '-'.join(key.split('-')[:2])
                # facet = self.qulac[self.qulac['topic_facet_id'] == topic_facet_id]['facet_desc']
                # facet = facet.iloc[0].replace('\\', '')

                # embed()

        elif self.hparams.history_data and self.hparams.human_data:
            self.df = pd.read_csv(self.hparams.history_data, sep='\t')
            np.random.seed(42)

            all_topics = sorted(self.df.topic_id.unique())
            topics = dict()
            topics['train'] = set(np.random.choice(all_topics, 50, replace=False))
            tmp = set(all_topics) - topics['train']
            topics['dev'] = set(np.random.choice(sorted(tmp), 11, replace=False))
            topics['test'] = topics['dev']

            self.df = self.df[self.df.topic_id.isin(topics[mode])]
            print(self.df.shape)
            self.df = self.df[~self.df.question.isna()]
            print(self.df.shape)

            ## add new Moh's crowdsourced answers
            df2 = pd.read_csv('../data/Moh_question_cases_answered.csv')

            train2 = pd.DataFrame()
            dev2 = pd.DataFrame()
            for case in df2.question2_case.unique():
                tmp = df2[df2['question2_case'] == case]
                dev_rows = np.random.choice(tmp.index, int(len(tmp) * 0.1), replace=False)
                train_rows = set(tmp.index) - set(dev_rows)
                train2 = train2.append(tmp.loc[list(train_rows)])
                dev2 = dev2.append(tmp.loc[dev_rows])

            if mode == 'train':
                self.df = self.df.append(train2)
            else:
                self.df = self.df.append(dev2)
            print(mode, self.df.shape)

        else:
            # self.df = pd.read_csv(os.path.join(self.data_dir, f'ClariQ_{mode}_yesno2.tsv'), sep='\t')
            # print(len(self.df))
            # self.df = self.df.dropna()
            # print(len(self.df))

            np.random.seed(42)
            self.df = pd.read_json(self.hparams.qulac_json)

            print(len(self.df))
            self.df = self.df.dropna()
            all_topics = sorted(self.df.topic_id.unique())
            topics = dict()
            topics['train'] = set(np.random.choice(all_topics, 150, replace=False))
            tmp = set(all_topics) - topics['train']
            topics['dev'] = set(np.random.choice(sorted(tmp), 23, replace=False))
            topics['test'] = topics['dev']
            self.df = self.df[self.df.topic_id.isin(topics[mode])]
            self.df = self.df.rename(columns={'topic': 'initial_request'})
            print(len(self.df))




    def __len__(self):

        if self.hparams.history_data and not self.hparams.human_data:
            return len(self.data)
        return len(self.df)
    
    def __getitem__(self, idx):
        # do not pad & create batches, just build_build_input_from_segments
        if self.mode == 'test':
            if self.hparams.answer_generation and not self.hparams.history_data:
                return self.test_mode_to_tensor_answer(idx)
            elif self.hparams.history_data and self.hparams.human_data:
                return self.test_mode_to_tensor_human_history(idx)
            elif self.hparams.history_data:
                return self.test_mode_to_tensor_history(idx)
            return self.test_mode_to_tensor(idx)

        if self.hparams.answer_generation and not self.hparams.history_data:
            return self.example_to_tensor_answer(idx)
        elif self.hparams.history_data and not self.hparams.human_data:
            return self.example_to_tensor_history(idx)
        elif self.hparams.history_data and self.hparams.human_data:
            return self.example_to_tensor_human_history(idx)
        return self.example_to_tensor(idx)

    def example_to_tensor_human_history(self, idx):
        sample = self.df.iloc[idx] 
        q = sample['query']
        cq = sample['question']
        ans = sample['answer']
        hist = ast.literal_eval(sample['history'])
        facet = sample['facet_desc']

        instance = self.build_input_with_history(facet, q, hist, cq,
                                        self.tokenizer, answer=ans,
                                        lm_labels=True, with_eos=True)
       
        input_seq = instance['input_ids']
        segments = instance['token_type_ids']
        mask = instance['attention_mask']
        target = instance['lm_label']

        # padding
        if len(input_seq) > self.hparams.max_seq_len:
            input_seq = input_seq[:self.hparams.max_seq_len]
            target = target[:self.hparams.max_seq_len]
            segments = segments[:self.hparams.max_seq_len]
            mask = mask[:self.hparams.max_seq_len]
        else:
            pad_num = self.hparams.max_seq_len - len(input_seq)
            input_seq.extend([self.tokenizer.pad_token_id] * pad_num)
            # Target should be padded with -100s
            target.extend([-100] * pad_num)
            segments.extend([self.tokenizer.pad_token_id] * pad_num)
            mask.extend([0] * pad_num)

        # distractor
        distractor = sample['distractor']
        mc_label = 0
        
        instance = self.build_input_with_history(facet, q, hist, cq,
                                        self.tokenizer, answer=distractor,
                                        lm_labels=True, with_eos=True)

        d_input_seq = instance['input_ids']
        d_segments = instance['token_type_ids']
        d_mask = instance['attention_mask']
        d_target = instance['lm_label']

        if len(d_input_seq) > self.hparams.max_seq_len:
            d_input_seq = d_input_seq[:self.hparams.max_seq_len]
            d_target = d_target[:self.hparams.max_seq_len]
            d_segments = d_segments[:self.hparams.max_seq_len]
            d_mask = d_mask[:self.hparams.max_seq_len]
        else:
            pad_num = self.hparams.max_seq_len - len(d_input_seq)
            d_input_seq.extend([self.tokenizer.pad_token_id] * pad_num)
            # Target should be padded with -100s
            d_target.extend([-100] * pad_num)
            d_segments.extend([self.tokenizer.pad_token_id] * pad_num)
            d_mask.extend([0] * pad_num)

        ret = {}
        ret['input_ids'] = torch.LongTensor([[input_seq, d_input_seq]])
        ret['lm_label'] = torch.LongTensor([[target, d_target]])
        # dialogue state embeddings
        ret['token_type_ids'] = torch.LongTensor([[segments, d_segments]])
        # attention mask to distinguish padding and real text
        ret['attention_mask'] = torch.LongTensor([[mask, d_mask]])
        ret['mc_label'] = torch.LongTensor([mc_label])

        return ret

    def example_to_tensor_history(self, idx):
        """ Returns encoded
        <facet_description> [SEP] <initial_request> [system] <clarifying_question_i>
        [user] <answer_i> 
        i is from the 1st turn to the last;
        LM label is the last answer in the conv."""

        key = self.idx_to_key[idx]
        sample = self.data[key] 
        q = sample['query']
        cq = sample['question']
        ans = sample['answer']
        hist = sample['history_list']
        # extract facets by key id
        topic_facet_id = '-'.join(key.split('-')[:2])
        facet = self.qulac[self.qulac['topic_facet_id'] == topic_facet_id]['facet_desc']
        facet = facet.iloc[0].replace('\\', '')

        instance = self.build_input_with_history(facet, q, hist, cq,
                                        self.tokenizer, answer=ans,
                                        lm_labels=True, with_eos=True)
       
        input_seq = instance['input_ids']
        segments = instance['token_type_ids']
        mask = instance['attention_mask']
        target = instance['lm_label']

        # padding
        if len(input_seq) > self.hparams.max_seq_len:
            input_seq = input_seq[:self.hparams.max_seq_len]
            target = target[:self.hparams.max_seq_len]
            segments = segments[:self.hparams.max_seq_len]
            mask = mask[:self.hparams.max_seq_len]
        else:
            pad_num = self.hparams.max_seq_len - len(input_seq)
            input_seq.extend([self.tokenizer.pad_token_id] * pad_num)
            # Target should be padded with -100s
            target.extend([-100] * pad_num)
            segments.extend([self.tokenizer.pad_token_id] * pad_num)
            mask.extend([0] * pad_num)

        # distractor
        distractor = ''
        while not distractor:
            n = np.random.randint(0, len(self.data))
            distractor = self.data[self.idx_to_key[n]]['answer']
            # distractor = self.qulac.sample(1).answer.values[0]
        mc_label = 0
        
        instance = self.build_input_with_history(facet, q, hist, cq,
                                        self.tokenizer, answer=distractor,
                                        lm_labels=True, with_eos=True)

        d_input_seq = instance['input_ids']
        d_segments = instance['token_type_ids']
        d_mask = instance['attention_mask']
        d_target = instance['lm_label']

        if len(d_input_seq) > self.hparams.max_seq_len:
            d_input_seq = d_input_seq[:self.hparams.max_seq_len]
            d_target = d_target[:self.hparams.max_seq_len]
            d_segments = d_segments[:self.hparams.max_seq_len]
            d_mask = d_mask[:self.hparams.max_seq_len]
        else:
            pad_num = self.hparams.max_seq_len - len(d_input_seq)
            d_input_seq.extend([self.tokenizer.pad_token_id] * pad_num)
            # Target should be padded with -100s
            d_target.extend([-100] * pad_num)
            d_segments.extend([self.tokenizer.pad_token_id] * pad_num)
            d_mask.extend([0] * pad_num)

        ret = {}
        ret['input_ids'] = torch.LongTensor([[input_seq, d_input_seq]])
        ret['lm_label'] = torch.LongTensor([[target, d_target]])
        # dialogue state embeddings
        ret['token_type_ids'] = torch.LongTensor([[segments, d_segments]])
        # attention mask to distinguish padding and real text
        ret['attention_mask'] = torch.LongTensor([[mask, d_mask]])
        ret['mc_label'] = torch.LongTensor([mc_label])

        return ret


    def example_to_tensor(self, idx):
        """ Returns encoded 
            <facets_descriptions> <conv_history> <clarifying_question>
            LM label is <clarifying_question>."""

        sample = self.df.iloc[idx]
        q = sample.initial_request
        cq = sample.question
        # ans = sample.answer
        facet = sample.facet_desc

        instance = self.build_input_from_segments(facet, q, cq, self.tokenizer, 
                                            lm_labels=True, with_eos=True,
                                            without_facets=self.hparams.without_facets)

        input_seq = instance['input_ids']
        segments = instance['token_type_ids']
        mask = instance['attention_mask']
        target = instance['lm_label']

        if len(input_seq) > self.hparams.max_seq_len:
            input_seq = input_seq[:self.hparams.max_seq_len]
            target = target[:self.hparams.max_seq_len]
            segments = segments[:self.hparams.max_seq_len]
            mask = mask[:self.hparams.max_seq_len]
        else:
            pad_num = self.hparams.max_seq_len - len(input_seq)
            input_seq.extend([self.tokenizer.pad_token_id] * pad_num)
            # Target should be padded with -100s
            target.extend([-100] * pad_num)
            segments.extend([self.tokenizer.pad_token_id] * pad_num)
            mask.extend([0] * pad_num)

        ret = {}
        ret['input_ids'] = torch.LongTensor(input_seq)
        ret['lm_label'] = torch.LongTensor(target)
        # dialogue state embeddings
        ret['token_type_ids'] = torch.LongTensor(segments)
        # attention mask to distinguish padding and real text
        ret['attention_mask'] = torch.LongTensor(mask)

        return ret

    def test_mode_to_tensor_human_history(self, idx):
        """ Returns encoded 
            <facets_descriptions> <conv_history> <clarifying_question> <answer>
            LM label is <clarifying_question>."""

        sample = self.df.iloc[idx] 
        q = sample['query']
        cq = sample['question']
        ans = sample['answer']
        hist = ast.literal_eval(sample['history'])
        facet = sample['facet_desc']
        mc_label = 0

        return {'facets': facet, 'query': q, 'history': hist,
                'question': cq, 'mc_label': mc_label}

    def test_mode_to_tensor_history(self, idx):
        """ Returns encoded 
            <facets_descriptions> <conv_history> <clarifying_question> <answer>
            LM label is <clarifying_question>."""

        key = self.idx_to_key[idx]
        sample = self.data[key] 
        q = sample['query']
        cq = sample['question']
        # ans = sample['answer']
        hist = sample['history_list']
        # extract facets by key id
        topic_facet_id = '-'.join(key.split('-')[:2])
        facet = self.qulac[self.qulac['topic_facet_id'] == topic_facet_id]['facet_desc']
        facet = facet.iloc[0].replace('\\', '')

        mc_label = 0

        return {'facets': facet, 'query': q, 'history': hist,
                'question': cq, 'mc_label': mc_label}

    def test_mode_to_tensor_answer(self, idx):
        """ Returns encoded 
            <facets_descriptions> <conv_history> <clarifying_question> <answer>
            LM label is <clarifying_question>."""

        sample = self.df.iloc[idx]
        q = sample.initial_request
        facet = sample.facet_desc
        question = sample.question
        # mc_label = sample.label
        mc_label = 0

        return {'facets': facet, 'history': q, 'question': question, 'mc_label': mc_label}


    def test_mode_to_tensor(self, idx):
        """ Returns encoded 
            <facets_descriptions> <conv_history> <clarifying_question>
            LM label is <clarifying_question>."""

        sample = self.df.iloc[idx]
        q = sample.initial_request
        facet = sample.facet_desc

        return {'facets': facet, 'history': q}

    def example_to_tensor_answer(self, idx):
        """ Returns encoded 
            <facets_descriptions> <conv_history> <clarifying_question> <answer>
            LM label is <answer>."""

        sample = self.df.iloc[idx]
        q = sample.initial_request
        cq = sample.question
        ans = sample.answer
        distractor = sample.distractor
        facet = sample.facet_desc
        # mc_label = sample.label
        mc_label = 0

        instance = self.build_input_from_segments(facet, q, cq, self.tokenizer, answer=ans,  
                                            lm_labels=True, with_eos=True,
                                            without_facets=self.hparams.without_facets)

        input_seq = instance['input_ids']
        segments = instance['token_type_ids']
        mask = instance['attention_mask']
        target = instance['lm_label']

        # padding
        if len(input_seq) > self.hparams.max_seq_len:
            input_seq = input_seq[:self.hparams.max_seq_len]
            target = target[:self.hparams.max_seq_len]
            segments = segments[:self.hparams.max_seq_len]
            mask = mask[:self.hparams.max_seq_len]
        else:
            pad_num = self.hparams.max_seq_len - len(input_seq)
            input_seq.extend([self.tokenizer.pad_token_id] * pad_num)
            # Target should be padded with -100s
            target.extend([-100] * pad_num)
            segments.extend([self.tokenizer.pad_token_id] * pad_num)
            mask.extend([0] * pad_num)

        instance = self.build_input_from_segments(facet, q, cq, self.tokenizer, answer=distractor,  
                                            lm_labels=True, with_eos=True,
                                            without_facets=self.hparams.without_facets)

        # distractor
        d_input_seq = instance['input_ids']
        d_segments = instance['token_type_ids']
        d_mask = instance['attention_mask']
        d_target = instance['lm_label']

        if len(d_input_seq) > self.hparams.max_seq_len:
            d_input_seq = d_input_seq[:self.hparams.max_seq_len]
            d_target = d_target[:self.hparams.max_seq_len]
            d_segments = d_segments[:self.hparams.max_seq_len]
            d_mask = d_mask[:self.hparams.max_seq_len]
        else:
            pad_num = self.hparams.max_seq_len - len(d_input_seq)
            d_input_seq.extend([self.tokenizer.pad_token_id] * pad_num)
            # Target should be padded with -100s
            d_target.extend([-100] * pad_num)
            d_segments.extend([self.tokenizer.pad_token_id] * pad_num)
            d_mask.extend([0] * pad_num)

        ret = {}
        ret['input_ids'] = torch.LongTensor([[input_seq, d_input_seq]])
        ret['lm_label'] = torch.LongTensor([[target, d_target]])
        # dialogue state embeddings
        ret['token_type_ids'] = torch.LongTensor([[segments, d_segments]])
        # attention mask to distinguish padding and real text
        ret['attention_mask'] = torch.LongTensor([[mask, d_mask]])
        ret['mc_label'] = torch.LongTensor([mc_label])

        return ret


    @staticmethod
    def build_input_with_history(facet, query, history, question, tokenizer, answer=None, lm_labels=False, with_eos=True):
        """ Build an input sequence from facets description, query, conv. history, clarifying question, and answer."""
        segment_id = 0

        input_seq, segments = [], []


        # adding facet description
        # input_seq.append(tokenizer.bos_token_id)
        input_seq += tokenizer.encode(facet)
        segments += [segment_id] * len(input_seq)
        segment_id += 1

        # add initial query
        # input_seq.append(tokenizer.sep_token_id)
        input_seq += tokenizer.encode("<user>")
        input_seq += tokenizer.encode(query)

        # adding conv history (or just initial query for first turn)
        for h in history:
            input_seq += tokenizer.encode("<system>")
            if isinstance(h['question'], list):
                input_seq += tokenizer.encode(h['question'][0])
            else:
                input_seq += tokenizer.encode(h['question'][0])
            input_seq += tokenizer.encode("<user>")
            if isinstance(h['answer'], list):
                input_seq += tokenizer.encode(h['answer'][0])
            else:
                input_seq += tokenizer.encode(h['answer'])

        # add current question
        input_seq += tokenizer.encode("<system>")
        input_seq += tokenizer.encode(question)
        segments += [segment_id] * (len(input_seq) - len(segments))
        segment_id += 1

        # sep token or <user>?
        # input_seq.append(tokenizer.sep_token_id)
        # input_seq += tokenizer.encode("<user>")
        # input_seq += tokenizer.encode(question)
        # segments += [segment_id] * (len(input_seq) - len(segments))
        # segment_id += 1


        input_seq.append(tokenizer.bos_token_id)
        # input_seq += tokenizer.encode("<user>")
        target_raw = answer

        if isinstance(target_raw, str):
            tmp = tokenizer.encode(target_raw)
        else:
            tmp = target_raw
        if with_eos:
            tmp.append(tokenizer.eos_token_id)

        if lm_labels:
            target = [-100] * len(input_seq)
            target += tmp 
        input_seq += tmp

        segments += [segment_id] * (len(input_seq) - len(segments))
        segment_id += 1
        mask = [1] * len(input_seq)

        instance = {}
        instance['input_ids'] = input_seq
        instance['lm_label'] = target if lm_labels else []
        # dialogue state embeddings
        instance['token_type_ids'] = segments
        # attention mask to distinguish padding and real text
        instance['attention_mask'] = mask

        return instance

    @staticmethod
    def build_input_from_segments(facets, history, question, tokenizer, answer=None, lm_labels=False, with_eos=True, without_facets=False):
        """ Build an input sequence from facets descriptions, conv. history, and the clarifying question."""
        segment_id = 0

        input_seq, segments = [], []
        if without_facets: # used for query-only baseline
            input_seq += tokenizer.encode(history)
            segments += [segment_id] * (len(input_seq) - len(segments))
            segment_id += 1
            # input_seq.append(tokenizer.sep_token_id) # here, not in target
            input_seq.append(tokenizer.bos_token_id)

        else: # add everything <facet terms> [SEP] <query> [bos] ...
            # adding facet descriptions
            input_seq += tokenizer.encode(facets)
            segments += [segment_id] * len(input_seq)
            segment_id += 1

            # adding conv history (or just initial query for first turn)
            input_seq.append(tokenizer.sep_token_id)
            input_seq += tokenizer.encode(history)
            segments += [segment_id] * (len(input_seq) - len(segments))
            segment_id += 1
            # input_seq.append(tokenizer.sep_token_id) # here, not in target

            if answer is not None:
                input_seq.append(tokenizer.sep_token_id)
                input_seq += tokenizer.encode(question)
                segments += [segment_id] * (len(input_seq) - len(segments))
                segment_id += 1

            input_seq.append(tokenizer.bos_token_id)

        if answer is not None:
            target_raw = answer
        else:
            target_raw = question

        if isinstance(target_raw, str):
            tmp = tokenizer.encode(target_raw)
        else:
            tmp = target_raw
        if with_eos:
            tmp.append(tokenizer.eos_token_id)

        if lm_labels:
            target = [-100] * len(input_seq)
            target += tmp 
        input_seq += tmp

        segments += [segment_id] * (len(input_seq) - len(segments))
        segment_id += 1
        mask = [1] * len(input_seq)

        instance = {}
        instance['input_ids'] = input_seq
        instance['lm_label'] = target if lm_labels else []
        # dialogue state embeddings
        instance['token_type_ids'] = segments
        # attention mask to distinguish padding and real text
        instance['attention_mask'] = mask

        return instance

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        parser.add_argument('--data_dir', type=str, default='../../ClariQ/data/')
        parser.add_argument('--mode', type=str, default='dev')
        parser.add_argument('--max_seq_len', type=int, default=512)
        return parser

if __name__ == "__main__":
    main_arg_parser = argparse.ArgumentParser(description="ClariQ dataset")
    parser = ClariQDatasetDH.add_model_specific_args(main_arg_parser, os.getcwd())
    args = parser.parse_args()

    SPECIAL_TOKENS = {'pad_token': '<pad>',
                      'sep_token': '<sep>',
                      'bos_token': '<bos>',
                      'eos_token': '<eos>'}

    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    cd = ClariQDatasetDH(tokenizer, args)

    model = GPT2LMHeadModel.from_pretrained('distilgpt2')
    model.resize_token_embeddings(len(tokenizer))

    X, Y, segments = [], [], []
    for i in range(5):
        xs = cd[i]
        X.append(xs['input_seq'])
        Y.append(xs['target'])
        segments.append(xs['token_type_ids'])

    X = torch.stack(X).to('cuda')
    Y = torch.stack(Y).to('cuda')
    segments = torch.stack(segments).to('cuda')

    model.to('cuda')

    out = model(X, token_type_ids=segments,
                labels=Y)
    loss = out[0]
    loss.backward()

