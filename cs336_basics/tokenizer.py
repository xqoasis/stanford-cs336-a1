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
    vocab = {}
    token_id = 0
    
    # add all possible bytes to vocab
    for i in range(256):
        vocab[token_id] = bytes([i])
        token_id += 1
    
    # add special tokens
    if special_tokens:
        for special_token in special_tokens:
            vocab[token_id] = special_token.encode('utf-8')
            token_id += 1
    
    # 3. convert text to byte sequence, then to token sequence
    # each byte is an initial token
    text_bytes = text.encode('utf-8')
    tokens = [bytes([b]) for b in text_bytes]
    
    # 4. BPE training: iterate to merge the most frequent token pairs
    merges = []
    
    # calculate how many merges are needed
    # vocab_size = 256 + len(special_tokens) + num_merges
    num_special = len(special_tokens) if special_tokens else 0
    target_merges = vocab_size - 256 - num_special
    
    # 5. merge tokens
    for i in range(target_merges):
        # count frequency of all adj token pairs and find most freq, add to merges
        pair_counts = Counter(zip(tokens, tokens[1:]))
        pair_to_merge = max(pair_counts.items(), key=lambda x: x[1])
        merges.append(pair_to_merge)

        # add merged token to vocab
        merged_token = pair_to_merge[0] + pair_to_merge[1]
        vocab[token_id] = merged_token
        token_id += 1

        # merge all occurrences of this pair in tokens
        new_tokens = []
        i = 0
        while i < len(tokens):
            if (i < len(tokens) - 1 and tokens[i] == pair_to_merge[0] and tokens[i + 1] == pair_to_merge[1]):
                new_tokens.append(merged_token)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens

        # print progress
        if (i + 1) % 1000 == 0:
            print(f"completed {i + 1} merges")
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
    special_tokens_sorted = sorted(special_tokens, key=lambda x: -len(x))
    pattern = "|".join(re.escape(tok) for tok in special_tokens_sorted)
    parts = re.split('(' + pattern + ')', text)
    
    return [part for part in parts if part]  # Remove empty strings


def pretokenize(text: str, special_tokens: List[str]) -> List[bytes]:
    """
    Separate text into pretokens using GPT-2 style regex pattern
    Special tokens are independent pretokens
    """
    parts = split_by_special_tokens(text, special_tokens)
    
    # GPT-2 pretokenization pattern
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    tokens_list = []
    for part in parts:
        if part in special_tokens:
            # Keep special tokens as single pretokens
            spec_tok_bytes = part.encode('utf-8')
            tokens_list.append(spec_tok_bytes)
        else:
            # Apply regex pattern to split into pretokens
            str_tokens = regex.findall(PAT, part)
            part_tokens = [s.encode('utf-8') for s in str_tokens]
            tokens_list.extend(part_tokens)
    
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
        self.id_to_token = {i: token for i, token in vocab.items()}
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
                    tokens[i] == merge_pair[0] and 
                    tokens[i + 1] == merge_pair[1]):
                    # Apply this merge
                    merged_token = merge_pair[0] + merge_pair[1]
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            
            tokens = new_tokens
        
        # Convert to token IDs
        token_ids = []
        for token in tokens:
            if token in self.token_to_id:
                token_ids.append(self.token_to_id[token])
            else:
                # Fallback: split unknown token into bytes
                for byte_val in token:
                    byte_token = bytes([byte_val])
                    if byte_token in self.token_to_id:
                        token_ids.append(self.token_to_id[byte_token])
        
        return token_ids
    
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
        for pretoken in pretokens:
            token_ids = self._apply_bpe_to_pretoken(pretoken)
            all_token_ids.extend(token_ids)
        
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
        byte_sequence = b""
        
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token_bytes = self.id_to_token[token_id]
                byte_sequence += token_bytes
            else:
                # handle unknown token id
                print(f"warning: unknown token id {token_id}")
                continue
        
        # decode byte sequence to string
        try:
            text = byte_sequence.decode('utf-8')
        except UnicodeDecodeError:
            # if decoding fails, use error handling strategy
            text = byte_sequence.decode('utf-8', errors='replace')
            print("warning: Unicode error during decoding")
        
        return text
    
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
            token_ids = self.encode(line)
            for token_id in token_ids:
                yield token_id


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