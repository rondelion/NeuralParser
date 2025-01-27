import argparse
import random

syntax = {}
lexicon = {}


def load_grammar(grammar):
    with open(grammar) as f:
        for s_line in f:
            s_line = s_line.strip()
            if s_line.find(':') > 0:
                buf = s_line.split(':')
                lhs = buf[0].strip()
                rhs = buf[1].split('+')
                rhs = list(map(lambda x: x.strip(), rhs))
                if lhs in syntax:
                    syntax[lhs].append(rhs)
                else:
                    syntax[lhs] = [rhs]
            elif s_line.find('-') > 0:
                buf = s_line.split('-')
                lhs = buf[0].strip()
                rhs = buf[1].split(',')
                rhs = list(map(lambda x: x.strip(), rhs))
                lexicon[lhs] = rhs

# print(syntax)
# print(lexicon)


def generate(buf, s, parentheses):
    ch = random.choice(buf)
    if len(ch) > 1:
        s = s + "{" if parentheses else s
    for cat in ch:
        if cat in lexicon:  # lex cat
            word = random.choice(lexicon[cat])
            s = s + word + " "
        else:
            s = generate(syntax[cat], s, parentheses)
    if len(ch) > 1:
        s = s.strip() + "} " if parentheses else s
    return s


def main():
    parser = argparse.ArgumentParser(description='Sentence generator with phrase structure grammar')
    parser.add_argument('--grammar', default="grammar.txt", help='grammar file path')
    parser.add_argument('--sentences', default="sentences.txt", help='sentence file path')
    parser.add_argument('--total', type=int, default=100, metavar='N',
                        help='Number of sentences (default: 100)')
    parser.add_argument('--parentheses', action='store_true', help='Specify iff you want parentheses {}')
    args = parser.parse_args()

    load_grammar(args.grammar)
    with open(args.sentences, mode='w') as f:
        for j in range(args.total):
            s = ""
            s = generate(syntax['S'], s, args.parentheses)
            f.write(s + '\n')


if __name__ == '__main__':
    main()
