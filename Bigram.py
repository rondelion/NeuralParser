import argparse
import os
import sys
import torch
from torch.utils.data import IterableDataset


class POSSequence(IterableDataset):
    def __init__(self, sentence_file, lexicon):
        super().__init__()
        self.lexicon = lexicon
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
                for i in range(wc):
                    word = words[i].strip()
                    pos = self.lexicon.word2pos[word]
                    yield pos
                yield 'EOS'


class Bigram:
    def __init__(self, lexicon):
        self.lexicon = lexicon
        self.pos_num = len(lexicon.pos2idx)
        self.prev_pos = "EOS"
        self.bigram = {}
        self.unigram = {}
        self.total = 0

    def eos(self):
        self.prev_pos = "EOS"

    def read_a_pos(self, pos):
        if self.prev_pos != "EOS" and pos != "EOS":
            bigram_str = self.prev_pos + ":" + pos
            if bigram_str in self.bigram:
                self.bigram[bigram_str] += 1
                self.unigram[self.prev_pos] += 1
            else:
                self.bigram[bigram_str] = 1
                if self.prev_pos in self.unigram:
                    self.unigram[self.prev_pos] += 1
                else:
                    self.unigram[self.prev_pos] = 1
            self.total += 1
        self.prev_pos = pos


class Lexicon:
    def __init__(self, grammar):
        self.word2pos = {}
        self.pos2idx = {}
        self.idx2pos = {}
        with open(grammar) as f:
            self.pos2idx['EOS'] = 0
            self.idx2pos[0] = 'EOS'
            idx = 1
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


def main():
    parser = argparse.ArgumentParser(description='Sentence parser')
    parser.add_argument('--sentences', default="sentences.txt", help='sentence file path')
    parser.add_argument('--grammar', default="grammar.txt", help='grammar file path')
    parser.add_argument('--output', default=None, help='bigram file')
    args = parser.parse_args()

    if os.path.isfile(args.grammar):
        lexicon = Lexicon(args.grammar)
    else:
        lexicon = None
        print("Grammar file " + "'" + args.grammar + "' is not found!")
        exit(-1)

    dataset = POSSequence(args.sentences, lexicon)
    data_loader = torch.utils.data.DataLoader(dataset)

    bg = Bigram(lexicon)
    for pos in data_loader:
        pos_str = pos[0]
        bg.read_a_pos(pos_str)
        if pos_str == 'EOS':
            bg.eos()

    if args.output is not None:
        f = open(args.output, mode='w')
    else:
        f = sys.stdout

    for key in bg.bigram:
        f.write(key + '\t' + str(bg.bigram[key]) + '\n')

    for key in bg.unigram:
        f.write(key + '\t' + str(bg.unigram[key]) + '\n')

    f.write("Total\t" + str(bg.total) + '\n')

    if args.output is not None:
        f.close()


if __name__ == '__main__':
    main()
