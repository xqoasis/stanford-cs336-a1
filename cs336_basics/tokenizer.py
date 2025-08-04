"""
BPE (Byte-Pair Encoding) Tokenizer Implementation

This module implements a BPE tokenizer from scratch, including:
1. train_bpe: Training BPE merges on a corpus  
2. Tokenizer: Encoding/decoding text using trained BPE model
"""

import os
import re
import regex  # For Unicode property classes like \p{L}
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional, Union


def train_bpe(
    corpus_path: Union[str, os.PathLike],
    vocab_size: int,
    special_tokens: List[str] = None
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    train BPE tokenizer
    
    Args:
        corpus_path: training corpus path
        vocab_size: target vocab size
        special_tokens: special token list, e.g. ['<unk>', '<pad>']
    
    Returns:
        vocab: {token_id: token_bytes} 词表映射
        merges: [(token1, token2), ...] 合并规则列表
    """
    # 1. read corpus
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 2. initialize vocab: all bytes(0-255) + special tokens
    vocab = {i: bytes([i]) for i in range(256)}
    if special_tokens:
        for special_token in special_tokens:
            vocab[len(vocab)] = special_token.encode('utf-8')
    
    # 3. Convert text to pretokens and build word frequencies
    pretokens = pretokenize(text, special_tokens or [])
    word_freqs = defaultdict(int)
    for pretoken_bytes in pretokens:
        # Skip special tokens - they shouldn't be split by BPE
        if special_tokens and pretoken_bytes.decode('utf-8', errors='ignore') in special_tokens:
            continue
        # Convert each pretoken to initial byte-level tokens as a tuple (for hashing)
        word_tokens = tuple(bytes([b]) for b in pretoken_bytes)
        if word_tokens:
            word_freqs[word_tokens] += 1

    # 4. BPE training: iterate to merge the most frequent token pairs
    merges = []
    target_merges = vocab_size - len(vocab)
    
    # 5. merge tokens
    for merge_step in range(target_merges):
        # count frequency of all adj token pairs and find most freq, add to merges
        pair_counts = defaultdict(int)
        for word_tokens, freq in word_freqs.items():
            for i in range(len(word_tokens) - 1):
                pair_counts[(word_tokens[i], word_tokens[i + 1])] += freq
        
        if not pair_counts:
            print(f"No more pairs to merge at step {merge_step}")
            break
            
        # Find most frequent pair (tie-breaking: lexicographically largest)
        max_freq = max(pair_counts.values())
        most_frequent_pair = max(pair for pair, freq in pair_counts.items() if freq == max_freq)
        merges.append(most_frequent_pair)

        # add merged token to vocab
        merged_token = most_frequent_pair[0] + most_frequent_pair[1]
        vocab[len(vocab)] = merged_token

        # update all words by merging this pair (i.e. update word_freqs)
        new_word_freqs = defaultdict(int)
        for word_tokens, freq in word_freqs.items():
            # Check if word contains the pair and merge it
            if any(word_tokens[i:i+2] == most_frequent_pair for i in range(len(word_tokens) - 1)):
                new_word = []
                i = 0
                while i < len(word_tokens):
                    if (i < len(word_tokens) - 1 and 
                        word_tokens[i:i+2] == most_frequent_pair):
                        new_word.append(merged_token)
                        i += 2
                    else:
                        new_word.append(word_tokens[i])
                        i += 1
                new_word_freqs[tuple(new_word)] += freq
            else:
                new_word_freqs[word_tokens] += freq
        
        word_freqs = new_word_freqs

        # print progress
        if (merge_step + 1) % 1000 == 0:
            print(f"Completed {merge_step + 1} merges")
    
    print(f"BPE training completed. Final vocab size: {len(vocab)}")
    return vocab, merges


def split_by_special_tokens(text: str, special_tokens: List[str]) -> List[str]:
    """
    Split on the special tokens
    example: 
        text = "Hello world! <|endoftext|> Great!" 
        special_tokens = ["<|endoftext|>"]
        result = ['Hello world! ', '<|endoftext|>', ' Great!']
    """
    if not special_tokens:
        return [text]
    
    # Sort by length (longest first) for greedy matching
    pattern = "|".join(re.escape(tok) for tok in sorted(special_tokens, key=len, reverse=True))
    return [part for part in re.split(f'({pattern})', text) if part]


def pretokenize(text: str, special_tokens: List[str]) -> List[bytes]:
    """
    Separate text into pretokens using GPT-2 style regex pattern
    Special tokens are independent pretokens
    """
    parts = split_by_special_tokens(text, special_tokens)
    
    # GPT-2 pretokenization pattern
    # Handle the english appr ('s, 'll, 've etc.)
    # Make sure the space and punctuation are not split
    # BPE for every part independently
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    tokens_list = []
    for part in parts:
        if part in special_tokens:
            # Keep special tokens as single pretokens
            tokens_list.append(part.encode('utf-8'))
        else:
            # Apply regex pattern to split into pretokens
            tokens_list.extend(s.encode('utf-8') for s in regex.findall(PAT, part))
    
    return tokens_list


class Tokenizer:
    """
    BPE Tokenizer class
    """
    
    def __init__(
        self, 
        vocab: Dict[int, bytes], 
        merges: List[Tuple[bytes, bytes]], 
        special_tokens: List[str] = None
    ):
        self.vocab = vocab  # {token_id: token_bytes}
        self.id_to_token = vocab
        self.token_to_id = {token: i for i, token in vocab.items()}
        
        self.merges = merges
        # Create a mapping of merge pairs to their ranks (higher rank = earlier merge)
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}
        
        self.special_tokens = special_tokens or []
        
        # create special token mapping
        self.special_token_ids = {}
        if special_tokens:
            for special_token in special_tokens:
                special_bytes = special_token.encode('utf-8')
                if special_bytes in self.token_to_id:
                    self.special_token_ids[special_token] = self.token_to_id[special_bytes]
    
    def _apply_bpe_to_pretoken(self, pretoken_bytes: bytes) -> List[int]:
        """
        Apply BPE merges to a single pretoken (byte sequence)
        Returns list of token IDs
        """
        if not pretoken_bytes:
            return []
        
        # Check if this is a special token
        if pretoken_bytes in self.token_to_id:
            return [self.token_to_id[pretoken_bytes]]
        
        # Convert bytes to initial token sequence
        tokens = [bytes([b]) for b in pretoken_bytes]
        
        # Apply BPE merges iteratively
        for merge_pair in self.merges:
            if len(tokens) < 2:
                break
                
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (i < len(tokens) - 1 and 
                    tokens[i] == merge_pair[0] and tokens[i + 1] == merge_pair[1]):
                    # Apply this merge
                    new_tokens.append(merge_pair[0] + merge_pair[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            
            tokens = new_tokens
        
        # Convert to token IDs
        return [self.token_to_id.get(token, self.token_to_id[bytes([token[0]])])
                for token in tokens if token in self.token_to_id]
    
    def encode(self, text: str) -> List[int]:
        """
        encode text to token id sequence
        
        Args:
            text: input text
            
        Returns:
            token id list
        """
        # 1. Pretokenize using GPT-2 style regex
        pretokens = pretokenize(text, self.special_tokens)
        
        # 2. Apply BPE to each pretoken independently
        all_token_ids = []
        for pretoken_bytes in pretokens:
            # Handle special tokens
            try:
                pretoken_str = pretoken_bytes.decode('utf-8')
                if self.special_tokens and pretoken_str in self.special_tokens:
                    # Special token - add its ID directly
                    if pretoken_str in self.special_token_ids:
                        all_token_ids.append(self.special_token_ids[pretoken_str])
                    continue
            except UnicodeDecodeError:
                pass  # Not a valid string, proceed with BPE
            
            all_token_ids.extend(self._apply_bpe_to_pretoken(pretoken_bytes))
        
        return all_token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """
        decode token id sequence to text
        
        Args:
            token_ids: token id list
            
        Returns:
            decoded text string
        """
        # convert token id to byte sequence
        byte_sequence = b''.join(self.id_to_token.get(token_id, b'') for token_id in token_ids)
        return byte_sequence.decode('utf-8', errors='replace')
    
    def get_vocab_size(self) -> int:
        """return vocab size"""
        return len(self.vocab)
    
    def get_vocab(self) -> Dict[int, bytes]:
        """return vocab"""
        return self.vocab.copy()
    
    def get_merges(self) -> List[Tuple[bytes, bytes]]:
        """return merge rules"""
        return self.merges.copy()
    
    def encode_iterable(self, text_iterable):
        """
        provide a generator for encoding large files line by line
        
        Args:
            text_iterable: iterable text object (e.g. file object)
            
        Yields:
            int: yield token id one by one
        """
        for line in text_iterable:
            yield from self.encode(line)


# test function
def test_tokenizer():
    """simple test function"""
    print("testing BPE Tokenizer...")
    
    # create test text
    test_text = "hello world! this is a test."
    
    # simulate small scale training (should use large corpus in practice)
    with open("/tmp/test_corpus.txt", "w", encoding="utf-8") as f:
        f.write(test_text * 100)  # repeat text to produce enough statistics
    
    # train BPE
    vocab, merges = train_bpe("/tmp/test_corpus.txt", vocab_size=300)
    
    # create tokenizer
    tokenizer = Tokenizer(vocab, merges)
    
    # test encode/decode
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"original text: {test_text}")
    print(f"encoded: {encoded}")
    print(f"decoded: {decoded}")
    print(f"vocab size: {tokenizer.get_vocab_size()}")
    print(f"merge count: {len(merges)}")
    
    # clean up
    os.remove("/tmp/test_corpus.txt")


if __name__ == "__main__":
    test_tokenizer()