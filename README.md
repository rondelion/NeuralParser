# NeuralParser
A parser without grammar (unsupervised parser) with a neural sequence memory.  
It parses a sequence of parts of speech into a tree.

## How to Install
* Clone the repository
* Clone [SequenceMemory](https://github.com/rondelion/SequenceMemory)
* Clone [AEPredictor](https://github.com/rondelion/AEPredictor) (uses SimplePredictor only)
* Install numpy
* Install pytorch

## Programs
* __`SentenceGenerator.py`__: generates sentences based on context free grammar (see [__`grammar.txt`__](https://github.com/rondelion/NeuralParser/blob/main/grammar.txt) as sample).
* __`NeuralParser.py`__: parses input text.
* __`Bigram.py`__: generates a statistics file for the modes P and O of NeuralParser.

## Usage
### Preparation
* Write a context free grammar (see [the sample](https://github.com/rondelion/NeuralParser/blob/main/grammar.txt)).
* Generate a sentence file (for training) with __`SentenceGenerator.py`__ and the grammar file.
* Generate a statistics file for the modes P and O of NeuralParser (optional).
* Generate sentence files for testing (optional).

### Running __`NeuralParser.py`__
#### Options
      --sentences: input sentence file
      --grammar: grammar file
      --config: configuration file
      --epochs: training epochs
      --mode:  N:neural, O:bigram occ., P: bigram prob., R: random (parse mode: see below)
      --next_pos: next pos predictor model file
      --cboc: cboc predictor model file
      --bigram: statistics file
      --adjoin: adjoins input sentences to the output parse if specified
      --output: output file (stdio if not specified)

#### Parse modes

     N: Neural: uses a neural next POS predictor in parsing.  
     O: bigram Occurrence: uses bigram occurrences in parsing (it uses the statics file).  
     P: bigram Probability: uses bigram conditional probabilities in parsing (it uses the statics file).  
     R: Random: uses random values in parsing.
      
#### Training and model files

NeuralParser uses two neural models.  

- __CBOC (Continuous Bag of Categories) model__: it is like the CBOW model in Word2Vec.  It uses parts of speech (POS) instead of words.  It is required in all the modes above.  
If no CBOC file is found, then the program trains the CBOC model with the given sentence file and epochs, and save it to a file.  
If a CBOC file is found, it loads the model from the file.
- __Next POS predictor model__: it predicts the next part of speech (POS).  It is required for the mode N.  
If the mode is N and no next POS predictor model file is found, then the program trains the model with the given sentence file and epochs, and save it to a file.  
If the mode is N and a next POS predictor model file is found, it loads the model from the file.




#### Sample usage
```
$ python NeuralParser.py --sentences sentences.txt --epochs 20 --mode N --adjoin --output outfile.txt

### Evaluation

You can use an edit distance tool such as [apted](https://pypi.org/project/apted/) for evaluation.  
The adjoin option yields the output format for apted so that you can apply apted to the output file with [the shell script](https://github.com/rondelion/NeuralParser/blob/main/apted.sh) and the total distance can be summed up with [sum.awk](https://github.com/rondelion/NeuralParser/blob/main/sum.awk).

