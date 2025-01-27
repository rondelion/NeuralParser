import argparse
import os
import sys
import numpy as np
import json

import torch
from torch.utils.data import IterableDataset

from SequenceMemory import OneHotDial
from SequenceMemory import SequenceMemory

from SimplePredictor import SimplePredictor


class POSSequence(IterableDataset):
    def __init__(self, sentence_file, lexicon):
        super().__init__()
        self.lexicon = lexicon
        self.vector_size = len(lexicon.pos2idx)
        self.eos_idx = self.lexicon.pos2idx['EOS']
        self.bos_idx = self.lexicon.pos2idx['BOS']
        self.eos_vector = np.zeros(self.vector_size, dtype=np.float32)
        self.eos_vector[self.eos_idx] = 1.0
        self.bos_vector = np.zeros(self.vector_size, dtype=np.float32)
        self.bos_vector[self.bos_idx] = 1.0
        if os.path.isfile(sentence_file):
            self.sentence_file = sentence_file
        else:
            print("Sentence file " + "'" + sentence_file + "' is not found!")
            exit(-1)

    def __iter__(self):
        with open(self.sentence_file, 'r') as f:
            for sentence in f:
                sentence = sentence.replace('{', ' ').replace('}', ' ')
                words = sentence.split()
                wc = len(words)
                yield 'BOS', self.bos_idx, self.bos_vector
                for i in range(wc):
                    word = words[i].strip()
                    pos = self.lexicon.word2pos[word]
                    pos_vector = np.zeros(self.vector_size, dtype=np.float32)
                    pos_idx = self.lexicon.pos2idx[pos]
                    pos_vector[pos_idx] = 1.0
                    yield pos, pos_idx, pos_vector
                yield 'EOS', self.eos_idx, self.eos_vector


class NetworkMemory:
    def __init__(self, input_dim, mem_size):
        self.input_dim = input_dim
        self.mem_size = mem_size
        self.ohd = OneHotDial(self.mem_size)
        self.sm = SequenceMemory(self.ohd, self.input_dim)
        self.activation = np.zeros(self.mem_size, dtype=float)
        self.prev_links = np.zeros((self.mem_size, self.mem_size), dtype=float)
        self.next_links = np.zeros((self.mem_size, self.mem_size), dtype=float)
        self.head_links = np.zeros((self.mem_size, self.mem_size), dtype=float)
        self.tail_links = np.zeros((self.mem_size, self.mem_size), dtype=float)
        self.idx_list = []  # for debug

    def set_a_link(self, links, src, tgt):
        vector = np.zeros(self.mem_size, dtype=float)
        vector[tgt] = 1.0
        links[src] = vector
        return links

    def get_afar(self):
        self.ohd.decay = self.ohd.decay / 2
        self.ohd.transition = self.ohd.transition / 2
        i = self.ohd.make_afar_hot()    # for debug
        self.idx_list.append(i)
        return i

    @staticmethod
    def get_target(links, src):
        if np.max(links[src]) == 0.0:
            print('Warning: NetworkMemory.get_target max link == 0', file=sys.stderr)
        return np.argmax(links[src])

    def reset(self):
        self.ohd = OneHotDial(self.mem_size)
        self.sm = SequenceMemory(self.ohd, self.input_dim)
        self.activation = np.zeros(self.mem_size, dtype=float)
        self.prev_links = np.zeros((self.mem_size, self.mem_size), dtype=float)
        self.next_links = np.zeros((self.mem_size, self.mem_size), dtype=float)
        self.head_links = np.zeros((self.mem_size, self.mem_size), dtype=float)
        self.tail_links = np.zeros((self.mem_size, self.mem_size), dtype=float)
        self.idx_list = []  # for debug


class SecondLayer:
    def __init__(self, config, lexicon):
        self.threshold = config["threshold"]
        self.lexicon = lexicon
        self.pos_num = len(lexicon.pos2idx)
        self.unigram_size = self.pos_num
        if config['mode'] == "N":
            config['next_pos_predictor']['input_dim'] = self.unigram_size
            config['next_pos_predictor']['output_dim'] = self.unigram_size
            config['next_pos_predictor']['hidden_dim'] = self.unigram_size
            self.next_pos_predictor = SimplePredictor(config['next_pos_predictor'])
        elif config['mode'] == "O" or config['mode'] == "P":
            self.bigram = {}
        self.mode = config['mode']
        config['cboc_predictor']['input_dim'] = self.unigram_size * 2
        config['cboc_predictor']['output_dim'] = self.unigram_size
        config['cboc_predictor']['hidden_dim'] = self.unigram_size
        self.cboc_predictor = SimplePredictor(config['cboc_predictor'])
        self.prev_pos = None
        self.prev = None
        self.prev_pos_idx = -1
        self.prev_prev_pos = None
        self.prev_prev = None
        self.nm = NetworkMemory(self.unigram_size, config['wm_size'])
        self.eos_idx = self.lexicon.pos2idx['EOS']
        self.eos()
        self.init = -1
        self.prev_idx = -1
        self.POS_list = []
        self.POS_list2 = []
        self.rng = np.random.default_rng()

    def load_bigram(self, bigram_file):
        with open(bigram_file) as f:
            for s_line in f:
                buf = s_line.strip().split('\t')
                if len(buf) > 1:
                    self.bigram[buf[0]] = int(buf[1])

    def eos(self):
        self.prev_prev_pos = None
        self.nm.reset()
        self.init = -1
        self.prev_idx = -1
        self.POS_list = []
        self.POS_list2 = []

    def train(self, data_loader, loaded_next_pos_model, loaded_cboc_model):
        next_pos_predictor_loss_sum = 0.0
        cboc_predictor_loss_sum = 0.0
        cnt = 0
        for pos, pos_idx, current in data_loader:
            pos_str = pos[0]
            if self.mode == "N" and not loaded_next_pos_model:
                if pos_str != "BOS" and self.prev_pos != "BOS" and pos_str != "EOS":
                    next_pos_predictor_loss_sum += self.next_pos_predictor.learn(self.prev, current)
            if not loaded_cboc_model:
                if self.prev_prev_pos is not None:
                    context = torch.cat((self.prev_prev, current), 1)
                    cboc_predictor_loss_sum += self.cboc_predictor.learn(context, self.prev)
            self.prev_prev_pos = self.prev_pos
            self.prev_prev = self.prev
            self.prev_pos = pos_str
            self.prev_pos_idx = pos_idx
            self.prev = current
            cnt += 1
        if not loaded_next_pos_model:
            print("next_pos_predictor_loss", next_pos_predictor_loss_sum / cnt)
        if not loaded_cboc_model:
            print("cboc_predictor_loss", cboc_predictor_loss_sum / cnt)

    def read_a_pos(self, pos, pos_vector):
        if self.init < 0:
            h = self.nm.sm.ohd.make_afar_hot()  # init
            self.init = h
            self.nm.idx_list.append(h)
        else:
            h = self.nm.get_afar()
            self.nm.ohd.transition[self.prev_idx, h] = 1.0
            self.nm.ohd.reverse[h, self.prev_idx] = 1.0
            self.nm.activation[h] = self.node_activation(self.prev, pos_vector)
        self.nm.sm.memorize_features(pos_vector, h)
        self.prev_idx = h
        self.prev_pos = pos
        self.prev = pos_vector
        self.POS_list.append(pos)

    def node_activation(self, head, tail):
        head_pos = self.lexicon.idx2pos[np.argmax(head)]
        tail_pos = self.lexicon.idx2pos[np.argmax(tail)]
        if head_pos == "BOS" or tail_pos == "EOS":
            return 0.0
        if self.mode == "N":
            head = head.astype(np.float32)
            prediction = self.next_pos_predictor.predictor(torch.from_numpy(head).clone()).cpu().detach().numpy().copy()
            return np.dot(prediction, tail)
        elif self.mode == "O" or self.mode == "P":
            return self.bigram_prob(head, tail)
        elif self.mode == "R":
            return self.rng.random()

    def bigram_prob(self, head, tail):
        head_pos = self.lexicon.idx2pos[np.argmax(head)]
        tail_pos = self.lexicon.idx2pos[np.argmax(tail)]
        bigram_count = self.bigram.get(head_pos + ":" + tail_pos, 0)
        if self.mode == "O":
            return bigram_count / self.bigram['Total']
        else:   # mode == "P" ~ conditional probability
            if head_pos not in self.bigram:
                return 0.0
            else:
                return bigram_count / self.bigram[head_pos]

    def parse(self):
        while np.max(self.nm.activation) > self.threshold:
            i = np.argmax(self.nm.activation)
            prev_i = self.nm.ohd.get_previous(i)
            if prev_i >= 0:
                prev_prev_i = self.nm.ohd.get_previous(prev_i)
                prev_prev = self.nm.sm.retrieve_features(prev_prev_i)
                post_i = np.argmax(self.nm.ohd.transition[i])
                post = self.nm.sm.retrieve_features(post_i)
                context = np.concatenate((prev_prev, post)).astype(np.float32)
                new = self.cboc_predictor.predictor(torch.from_numpy(context).clone()).cpu().detach().numpy().copy()
                if np.argmin(self.nm.ohd.decay) in self.nm.idx_list:
                    print("Already used")
                new_i = self.nm.get_afar()
                self.nm.sm.memorize_features(new, new_i)
                if self.lexicon.idx2pos[np.argmax(prev_prev)] == "BOS":
                    self.nm.activation[new_i] = 0.0
                else:
                    self.nm.activation[new_i] = self.node_activation(prev_prev, new)
                if self.lexicon.idx2pos[np.argmax(post)] == "EOS":
                    self.nm.activation[post_i] = 0.0
                else:
                    self.nm.activation[post_i] = self.node_activation(new, post)
                self.nm.ohd.transition[prev_prev_i, new_i] = 1.0
                self.nm.ohd.reverse[new_i, prev_prev_i] = 1.0
                self.nm.ohd.transition[new_i, post_i] = 1.0
                self.nm.ohd.reverse[post_i, new_i] = 1.0
                self.nm.ohd.transition[prev_prev_i, prev_i] = 0.0
                self.nm.ohd.reverse[prev_i, prev_prev_i] = 0.0
                self.nm.ohd.transition[prev_i, i] = 0.0
                self.nm.ohd.reverse[i, prev_i] = 0.0
                self.nm.ohd.transition[i, post_i] = 0.0
                self.nm.ohd.reverse[post_i, i] = 0.0
                self.nm.set_a_link(self.nm.head_links, new_i, prev_i)
                self.nm.set_a_link(self.nm.tail_links, new_i, i)
                if self.init == prev_i:
                    self.init = new_i
                self.nm.activation[i] = 0.0
                self.nm.activation[prev_i] = 0.0

    def inspect_next_pos_predictor(self):
        size = len(self.lexicon.pos2idx)
        for i in range(size):
            for j in range(size):
                head = np.zeros(size).astype(np.float32)
                head[i] = 1
                tail = np.zeros(size)
                tail[j] = 1
                prediction = self.next_pos_predictor.predictor(
                    torch.from_numpy(head).clone()).cpu().detach().numpy().copy()
                dot = np.dot(prediction, tail)
                print(self.lexicon.idx2pos[i], self.lexicon.idx2pos[j], dot)

    def dump_parses(self, f, idx):
        while True:
            if np.max(self.nm.head_links[idx]) == 0:
                pos_feature = self.nm.sm.retrieve_features(idx)
                pos = self.lexicon.idx2pos[np.argmax(pos_feature)]
                if pos != 'BOS' and pos != 'EOS':
                    f.write(pos)
                self.POS_list2.append(pos)
            else:
                self.dump_tree(f, idx)
            if np.max(self.nm.sm.ohd.transition[idx]) == 0:
                if len(self.POS_list) != len(self.POS_list2):
                    print("Something wrong!")
                break
            else:
                idx = np.argmax(self.nm.sm.ohd.transition[idx])
                f.write(' ')
        # f.write('\n')

    def dump_tree(self, f, idx):
        f.write('{')
        # head
        head = np.argmax(self.nm.head_links[idx])
        if head == idx:
            print("Something wrong!")
            return
        if np.max(self.nm.head_links[head]) == 0:
            pos_feature = self.nm.sm.retrieve_features(head)
            pos = self.lexicon.idx2pos[np.argmax(pos_feature)]
            f.write(pos)
            self.POS_list2.append(pos)
        else:
            self.dump_tree(f, head)
        # tail
        if np.max(self.nm.tail_links[idx]) > 0:
            f.write(' ')
            tail = np.argmax(self.nm.tail_links[idx])
            if np.max(self.nm.head_links[tail]) == 0:
                pos_feature = self.nm.sm.retrieve_features(tail)
                pos = self.lexicon.idx2pos[np.argmax(pos_feature)]
                f.write(pos)
                self.POS_list2.append(pos)
            else:
                self.dump_tree(f, tail)
        f.write('}')


class Lexicon:
    def __init__(self, grammar):
        self.word2pos = {}
        self.pos2idx = {}
        self.idx2pos = {}
        with open(grammar) as f:
            self.pos2idx['EOS'] = 0
            self.idx2pos[0] = 'EOS'
            self.pos2idx['BOS'] = 1
            self.idx2pos[1] = 'BOS'
            idx = 2
            for s_line in f:
                s_line = s_line.strip()
                if s_line.find('-') > 0:
                    buf = s_line.split('-')
                    lhs = buf[0].strip()
                    if lhs not in self.pos2idx:
                        self.pos2idx[lhs] = idx
                        self.idx2pos[idx] = lhs
                        idx += 1
                    rhs = buf[1].split(',')
                    for word in rhs:
                        self.word2pos[word.strip()] = lhs


def replace2categories(x, lexicon):
    buf = ""
    word = ""
    for c in x:
        if c in '{} ':
            if word != "":
                buf = buf + lexicon.word2pos.get(word, "UNK")
                word = ""
            buf = buf + c
        else:
            word = word + c
    return buf


def insert_non_terminal(x):
    buf = ""
    terminal = ["", ""]
    idx = 0
    for c in x:
        if terminal[0] == "":
            if c == "{":
                buf = buf + "{X"
            elif c == "}":
                buf = buf + "}"
            elif c != " ":
                terminal[0] = c
        else:
            if c == " ":
                idx += 1
            elif c == "{":
                buf = buf + "{" + terminal[0] + "}{X"
                terminal = ["", ""]
                idx = 0
            elif c != "}":
                terminal[idx] += c
            else:       # }
                if terminal[1] == "":
                    buf = buf + "{" + terminal[0] + "}}"
                else:
                    buf = buf + "{{" + terminal[0] + "}{" + terminal[1] + "}}}"
                terminal = ["", ""]
                idx = 0
    return buf


def main():
    parser = argparse.ArgumentParser(description='Sentence parser')
    parser.add_argument('--sentences', default="sentences.txt", help='sentence file path')
    parser.add_argument('--grammar', default="grammar.txt", help='grammar file path')
    parser.add_argument('--config', type=str, default='NeuralParser.json', metavar='N',
                        help='Configuration (default: NeuralParser.json')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='Number of training epochs (default: 1)')
    parser.add_argument('--mode', help='O:bigram occ., P: bigram prob., N:neural, R: random')
    parser.add_argument('--next_pos', default="next_pos.pt", help='next pos predictor model file')
    parser.add_argument('--cboc', default="cboc.pt", help='cboc predictor model file')
    parser.add_argument('--bigram', default="bigram.txt", help='statistics file')
    parser.add_argument('--adjoin', action='store_true', help='Specify iff you want adjoin input to output')
    parser.add_argument('--output', help='output file')

    args = parser.parse_args()

    with open(args.config) as config_file:
        config = json.load(config_file)

    config['mode'] = args.mode

    if os.path.isfile(args.grammar):
        lexicon = Lexicon(args.grammar)
    else:
        lexicon = None
        print("Grammar file " + "'" + args.grammar + "' is not found!")
        exit(-1)

    dataset = POSSequence(args.sentences, lexicon)
    data_loader = torch.utils.data.DataLoader(dataset)

    sl = SecondLayer(config, lexicon)

    loaded_next_pos_model = False
    if args.mode == "N":
        if os.path.isfile(args.next_pos):
            print("Loading next pos model: " + args.next_pos)
            sl.next_pos_predictor.predictor.load_state_dict(torch.load(args.next_pos, weights_only=True))
            loaded_next_pos_model = True
    elif args.mode == "O" or args.mode == "P":
        sl.load_bigram(args.bigram)
        loaded_next_pos_model = True
    elif args.mode == "R":
        pass
    else:
        print('No mode specified!', file=sys.stderr)
        exit(-1)

    loaded_cboc_model = False
    if os.path.isfile(args.cboc):
        print("Loading CBOC model: " + args.cboc)
        sl.cboc_predictor.predictor.load_state_dict(torch.load(args.cboc, weights_only=True))
        loaded_cboc_model = True

    # Training Mode
    if not ((loaded_next_pos_model or args.mode == "R") and loaded_cboc_model):
        for epoch in range(args.epochs):
            print('epoch:', epoch + 1)
            sl.train(data_loader, loaded_next_pos_model, loaded_cboc_model)

    if args.mode == "N" and not loaded_next_pos_model:
        print("Saving next pos model: " + args.next_pos)
        torch.save(sl.next_pos_predictor.predictor.state_dict(), args.next_pos)

    if not loaded_cboc_model:
        print("Saving CBOC model: " + args.cboc)
        torch.save(sl.cboc_predictor.predictor.state_dict(), args.cboc)

    if args.adjoin:
        with open(args.sentences) as f:
            lines = f.readlines()

    import io
    out_f = open(args.output, mode='w') if args.output is not None else sys.stdout
    # Parse Mode
    counter = 0
    for pos, pos_idx, pos_vector in data_loader:
        pos = pos[0]
        pos_vector = pos_vector[0].cpu().detach().numpy().copy()
        sl.read_a_pos(pos, pos_vector)
        if pos == 'EOS':
            sl.parse()
            io_object = io.StringIO()
            sl.dump_parses(io_object, sl.init)
            if args.adjoin:
                out_f.write(insert_non_terminal(io_object.getvalue()) + '\t'
                            + insert_non_terminal(replace2categories(lines[counter], lexicon)) + '\n')
            else:
                out_f.write(io_object.getvalue() + '\n')
            sl.eos()
            counter += 1


if __name__ == '__main__':
    main()
