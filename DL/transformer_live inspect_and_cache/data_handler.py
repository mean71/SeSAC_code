"""
Data handling scripts. 

1. class Vocabulary: Vocabulary wrapper class for generating/handling special tokens/index to word, and vice versa, etc,... 
2. Class LanguagePair: Wrapper class for sentences of language pair. 
"""
import torch 
from collections import defaultdict 
from torch.utils.data import DataLoader, TensorDataset
from typing import Any, Dict, List, Tuple, Union

from util import inspect_and_cache, generate_histogram, select_by_coverage

develop = True

class Vocabulary:
    """
    Handles vocabulary of a given sequential dataset. 

    Args:
        coverage (float): Coverage for determining whether the token shall be considered as OOV or not. 
    Attributes:
        word2index (dict[str, int]): Dict containing token as key, and index of the token as value. 
        index2word (dict[int, str]): Dict containing index of token as key, and the token as value. 
        vocab_size (int): Integer representing the vocabulary size. 
    Methods:
        
    Variables: 
        SPECIAL_TOKENS
        EOS_IDX, EOS 
        SOS_IDX, SOS
        PAD_IDX, PAD 
        OOV_IDX, OOV 
    """
    EOS: str = '[EOS]'
    SOS: str = '[SOS]'
    PAD: str = '[PAD]'
    OOV: str = '[OOV]'
    SPECIAL_TOKENS: List[str] = [EOS, SOS, PAD, OOV]
    
    def __init__(self):
        self.word2index: Dict[str, int] = defaultdict(lambda : Vocabulary.SPECIAL_TOKENS.index(Vocabulary.OOV))
        self.index2word: Dict[int, str] = {}
        self.vocab_size: int = 0 
        self.word2freq: defaultdict[str, int] = defaultdict(int)

        self.eos_idx: int = Vocabulary.SPECIAL_TOKENS.index(Vocabulary.EOS)
        self.sos_idx: int = Vocabulary.SPECIAL_TOKENS.index(Vocabulary.SOS)
        self.pad_idx: int = Vocabulary.SPECIAL_TOKENS.index(Vocabulary.PAD)
        self.oov_idx: int = Vocabulary.SPECIAL_TOKENS.index(Vocabulary.OOV)


    def add_word(self, token: str):
        """
        Adds a token to the vocabulary if it doesn't exists. 
        If it exists, do nothing. 

        Args:
            token (str): The token to be added. 
        """
        self.word2freq[token] += 1 

    def finalize_vocabulary(self, coverage: float = 0.999):
        assert coverage <= 1 

        vocab_list: List[Tuple[str, int]] = [(t, -1) for t in Vocabulary.SPECIAL_TOKENS] + select_by_coverage(self.word2freq, coverage)

        self.vocab_size = len(vocab_list)
        
        for word_idx, (word, freq) in enumerate(vocab_list):
            if word not in self.word2index:
                self.word2index[word] = word_idx
                self.index2word[word_idx] = word 
        
        for token in Vocabulary.SPECIAL_TOKENS:
            assert self.word2index[token] == Vocabulary.SPECIAL_TOKENS.index(token), debug_shell()

        self.word2index = dict(self.word2index)
        self.index2word = dict(self.index2word) 
        self.word2freq = dict(self.word2freq)

    def word2idx(self, token):
        if token in self.word2index:
            return self.word2index[token] 
        else:
            return self.oov_idx 

    def sentence2tensor(self, 
            sentence: List[str], 
            tokenize_strategy: str = 'split', 
    ) -> torch.tensor:
        tokens: List[str] = tokenize(sentence, tokenize_strategy)

        return torch.tensor(
            [self.word2idx(token) for token in tokens],
            dtype = torch.long
        )

    def tensor2sentence(self, 
            tensor: torch.tensor, 
            clean: bool = False, 
    ) -> Union[List[str], str]:
        
        index_list = tensor.tolist()
        
        if clean:
            return ' '.join([self.index2word[e] for e in index_list if self.index2word[e] not in Vocabulary.SPECIAL_TOKENS])
        return [self.index2word[e] for e in index_list]

class LanguagePair:
    """
    Wrapper class containing language pair dataset. 

    Args:
        source_sentences (List[str]): Sentences from source language. 
        target_sentences (List[str]): Sentences from target langauge. 
        batch_size (int): Batch size for self.data containing source_sentences-target_sentences pair. 
    Attributes:
        source_sentences (List[List[str]]): List of list of tokens. Each list of tokens denote the source sentences, tokenized by LanguagePair.tokenize. 
        target_sentences (List[List[str]]): List of list of tokens. Each list of tokens denote the target sentences, tokenized by LanguagePair.tokenize. 
        source_vocab (Vocabulary): Instance of Vocabulary class constructed from source_sentences. 
        target_vocab (Vocabulary): Instance of Vocabulary class constructed from target_sentences. 
        data (torch.DataLoader): torch.Dataloader containing source-target mapped dataset. 
    Methods:
        static initiate_from_file (str data_file_path) -> LanguagePair 
            Generate a LanguagePair instance from the given file. 
            The file is assumed to have a source sentence, target sentence, and a license information seperated by a tab. 
            Example:
                This work isn't easy.	この仕事は簡単じゃない。	CC-BY 2.0 (France) Attribution: tatoeba.org #3737550 (CK) & #7977622 (Ninja)
                Those are sunflowers.	それはひまわりです。	CC-BY 2.0 (France) Attribution: tatoeba.org #441940 (CK) & #205407 (arnab)
                Tom bought a new car.	トムは新車を買った。	CC-BY 2.0 (France) Attribution: tatoeba.org #1026984 (CK) & #2733633 (tommy_san)
                This watch is broken.	この時計は壊れている。	CC-BY 2.0 (France) Attribution: tatoeba.org #58929 (CK) & #221604 (bunbuku)

    """
    def __init__(
            self, 
            source_sentences: List[str], 
            target_sentences: List[str], 
            filter_long_sequence: bool = True, 
            batch_size: int = 32, 
    ):
        assert len(source_sentences) == len(target_sentences)

        def preprocess_sentences(
                sents: List[str], 
                append_eos: bool = False, 
                append_sos: bool = False, 
        ) -> Tuple[Vocabulary, List[List[str]], int]:
            
            vocab: Vocabulary = Vocabulary()
            tokenized_sentences: List[List[str]] = []

            for sent in sents:
                tokens: List[str] = tokenize(sent) 
                if append_sos:
                    tokens = [Vocabulary.SOS] + tokens 
                if append_eos:
                    tokens.append(Vocabulary.EOS)
                
                for token in tokens:
                    vocab.add_word(token)

                tokenized_sentences.append(tokens)
            
            sentence_histogram: defaultdict[int, int] = generate_histogram(
                tokenized_sentences, 
                key = len,
            )

            max_seq_length: int = select_by_coverage(
                sentence_histogram, 
                key = lambda x: x[0],  
                reverse = False,
            )[-1][0]

            vocab.finalize_vocabulary()
            
            return vocab, tokenized_sentences, max_seq_length

        def make_tensor(
                sent_list: List[List[str]], 
                vocab: Vocabulary, 
        ) -> torch.tensor: 
            max_sentence_length: int = max([len(e) for e in sent_list])
            res: torch.tensor = torch.zeros(len(sent_list), max_sentence_length, dtype = torch.long)
            
            for idx, sent in enumerate(sent_list):
                lst: List[int] = [vocab.word2idx(token) for token in sent] + [vocab.pad_idx for _ in range(max_sentence_length - len(sent))]
                assert len(lst) == max_sentence_length
                
                res[idx] = torch.tensor(lst, dtype = torch.long)
            
            return res 

        source_vocab, source_sentences, source_max_length = preprocess_sentences(source_sentences)

        target_vocab, target_sentences, target_max_length  = preprocess_sentences(target_sentences, append_eos = True, append_sos = True)

        sources = []
        targets = []

        for src, tgt in zip(source_sentences, target_sentences):
            if filter_long_sequence:
                if not (len(src) > source_max_length or len(tgt) > target_max_length):
                    sources.append(src) 
                    targets.append(tgt) 
            else:
                sources.append(src[:source_max_length])
                targets.append(tgt[:target_max_length])

        self.source_vocab: Vocabulary = source_vocab
        self.target_vocab: Vocabulary = target_vocab
        self.source_sentences = sources
        self.target_sentences = targets

        source_tensor = make_tensor(source_sentences, source_vocab) 
        target_tensor = make_tensor(target_sentences, target_vocab)

        dataset = TensorDataset(source_tensor, target_tensor)

        dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)

        self.data = dataloader 

    
    
@inspect_and_cache
def initiate_from_file(data_file_path: str) -> LanguagePair:
    """
    From the given string that contains the path to file, open the file, read, and split each lines by a delimeter. 
    """
    source_sentences = []
    target_sentences = []

    if develop:
        print(f'Initiated making LanguagePair instance from {data_file_path}')

    with open(data_file_path, 'r', encoding = 'utf-8') as f:
        for line in f.readlines():
            tab_seperated = line.strip().split('\t')

            if len(tab_seperated) == 2:
                source, target = tab_seperated
            else:
                source, target, *license_info = line.strip().split('\t')
            source_sentences.append(source)
            target_sentences.append(target)

    return LanguagePair(source_sentences, target_sentences)
    

def tokenize(sentence: str, tokenize_strategy: str = 'split') -> List[str]:
    """
    tokenize (str sentence, str tokenize_strategy) -> List[str]
        Tokenize the given sentences to list of tokens. 
        Tokenize the first argument sentence, based on the argument tokenize_strategy. 
        Each options will be discussed in the docstring of tokenize method. 

        Example:
            en2ko = LanguagePair.initiate_from_file('en2ko.txt')
            en2ko.tokenize('This work isn't easy.')
            >> ['this', 'work', 'isn`t', 'easy.'] 
    """
    if tokenize_strategy == 'split':
        return sentence.split()
    else:
        NotImplemented


if __name__ == '__main__':
    from config import en2fr_data
    from debugger import debug_shell 
    
    en2fr: LanguagePair = initiate_from_file(en2fr_data)
    
    debug_shell()

    