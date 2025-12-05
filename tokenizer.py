import json
import numpy as np
import torch
from typing import Union, List, Optional, Any, Dict
from transformers.utils import (
    ExplicitEnum,
    PaddingStrategy,
    PushToHubMixin,
    TensorType)
from transformers.tokenization_utils_base import (
    BatchEncoding
    , EncodedInput
)
import emoji
import json

class CharTokenizer:

    def __init__(self, charset_file_path, max_word_length: int = 12, word2id=None, charset: dict = None):
        if charset_file_path is None and charset is None:
            raise ValueError("Either charset_file_path or charset must be provided.")
        if charset_file_path is not None:
            charset = json.load(open(charset_file_path, 'r', encoding='utf-8'))
        
        self.valid_char2id = charset['valid_char2id']
        self.valid_id2char = charset['valid_id2char']
        # cast the keys to int
        self.valid_id2char = {int(k): v for k, v in self.valid_id2char.items()}
        self.special_char2name = charset['special_char2name']
        self.special_name2char = charset['special_name2char']
        self.special_token2char = charset['special_token2char']
        
        self.special_token2word = {k: v*max_word_length for k, v in self.special_token2char.items()}
        self.special_word2token = {v: k for k, v in self.special_token2word.items()}

        self.max_word_length = max_word_length
        self.model_input_names = ['input_ids', 'attention_mask']
        self.pad_char_id = self.valid_char2id[self.special_token2char['[PAD]']]

        self.set_word2id(word2id)
    
    @staticmethod
    def from_pretrained(pretrained_directory: str):
        """
        Load the tokenizer from a directory.
        """
        # read tokenizer_config.json
        with open(f"{pretrained_directory}/tokenizer_config.json", 'r', encoding='utf-8') as f:
            tokenizer_config = json.load(f)
        max_word_length = tokenizer_config['max_word_length']
        word2id = None
        charset = tokenizer_config['charset']
        return CharTokenizer(None, max_word_length=max_word_length, word2id=word2id, charset=charset)

    @property
    def char_vocab_size(self) -> int:
        """
        Returns the size of the character vocabulary.
        """
        return len(self.valid_char2id)

    def set_word2id(self, word2id: Dict[str, int]):
        """
        Set the word2id mapping.
        """
        self.word2id  = word2id
        if word2id:
            # assert special tokens are in word2id
            for token in self.special_token2word:
                if token not in self.word2id:
                    raise ValueError(f"Special token {token} not found in word2id.")
            self.id2word = {v: k for k, v in self.word2id.items()}
        
    @property
    def vocab_size(self) -> int:
        """
        Returns the size of the vocabulary.
        """
        return len(self.valid_char2id)

    def _encode_word(self, word: str) -> list:
        """
        Encode a word into a list of character IDs.
        """
        
        if word in self.special_token2word:
            word = self.special_token2word[word]
        else:
            word = word[:self.max_word_length-1]
            word = word + self.special_name2char['eos']
            word = word.ljust(self.max_word_length, self.special_name2char['wpad'])
        return [self.valid_char2id.get(char, self.valid_char2id[self.special_name2char['unk']]) for char in word]
    
    def clean_word(self, word: str) -> str:
        word = word[:self.max_word_length-1]
        return "".join([char if char in self.valid_char2id else self.special_name2char['unk'] for char in word])
    
    def clean_sentence(self, sentence: str) -> str:
        """
        Clean a sentence by replacing invalid characters with the unknown character.
        """
        return " ".join([self.clean_word(word) for word in sentence.split()])

    def _decode_word(self, ids: list, strip=False) -> str:
        """
        Decode a list of character IDs back into a word.
        """
        word = ''.join(self.valid_id2char.get(id, self.special_name2char['unk']) for id in ids)
        if word in self.special_word2token:
            return self.special_word2token[word]
        if strip:
            word = word.rstrip(self.special_name2char['pad'])
        return word
    
    def _encode_words(self, words: list) -> list:
        """
        Encode a list of words into a list of lists of character IDs.
        """
        return [self._encode_word(word) for word in words]
    
    def _decode_words(self, ids_list: list, strip=False) -> list:
        """
        Decode a list of lists of character IDs back into a list of words.
        """
        return [self._decode_word(ids, strip) for ids in ids_list]
    
    def _encode_with_special_tokens(self, words: list, word_pad_length: int = 0, add_cls: bool = False, add_sep: bool = False, max_length: int = None) -> list:
        """
        Encode a list of words with special tokens.
        """
        encoded_words = self._encode_words(words)
        if not max_length or type(max_length) is not int:
            max_length = len(encoded_words)
        else:
            if add_cls:
                max_length -= 1
            if add_sep:
                max_length -= 1
        assert max_length > 0, "max_length must be greater than 0"
        if len(encoded_words) >= max_length:
            encoded_words = encoded_words[:max_length]
        if add_cls:
            encoded_words.insert(0, self._encode_word('[CLS]'))
        
        if add_sep:
            encoded_words.append(self._encode_word('[SEP]'))
        return encoded_words
    
    def _encode(self, words: list, 
                    pad_to_length: int = 0, 
                    max_length:int=None, 
                    add_cls: bool = False, 
                    add_sep: bool = False) -> list:
        """
        Encode a list of words with special tokens and sequence padding.
        """
    
        return self._encode_with_special_tokens(words, pad_to_length, add_cls, add_sep, max_length=max_length)

    def _remove_chars_after_eos(self, word: str) -> str:
        """
        Remove characters after the end of sentence (eos) token.
        """
        if self.special_name2char['eos'] in word:
            return word.split(self.special_name2char['eos'])[0]
        return word

    def decode(self, input_ids: np.ndarray, skip_special_tokens: bool = True) -> List[str]:
        """
        Decode the output dictionary (as returned by encode) back into sentences.
        """
        if isinstance(input_ids, np.ndarray) or isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()
        sentences = []
        for word_ids in input_ids:
            words = self._decode_words(word_ids, strip=True)
            if skip_special_tokens:
                words = [w for w in words if w not in self.special_token2word]
            words = [self._remove_chars_after_eos(w) for w in words]
            sentence = " ".join(words)
            sentences.append(sentence)
        return sentences
    
    def encode_to_word_ids(self, words: list,
                          pad_to_length: int = 0,
                          max_length: int = None,
                          add_cls: bool = False,
                          add_sep: bool = False) -> list:
        """
        Encode a list of words into a list of word IDs.
        This method is similar to `_encode`, but it returns word IDs instead of character IDs.
        """
        encoded_words = [self.word2id[word] if word in self.word2id else self.word2id['[UNK]'] for word in words]
        if not max_length or type(max_length) is not int:
            max_length = len(encoded_words)
        else:
            if add_cls:
                max_length -= 1
            if add_sep:
                max_length -= 1
        assert max_length > 0, "max_length must be greater than 0"
        if len(encoded_words) >= max_length:
            encoded_words = encoded_words[:max_length]
        if add_cls:
            encoded_words.insert(0, self.word2id['[CLS]'])
        if add_sep:
            encoded_words.append(self.word2id['[SEP]'])
        if pad_to_length > 0:
            if len(encoded_words) < pad_to_length:
                encoded_words += [self.word2id['[PAD]']] * (pad_to_length - len(encoded_words))
            else:
                encoded_words = encoded_words[:pad_to_length]
        return encoded_words
    
    def deocde_to_words(self, word_ids: list[list], strip=True) -> list:
        """
        Decode a list of word IDs back into a list of words.
        This method is similar to `_decode_words`, but it takes word IDs instead of character IDs.
        """
        # if word_ids is list of int, convert it to list of list
        if isinstance(word_ids[0], int):
            word_ids = [word_ids]
        texts = []
        for ids in word_ids:
            words = [self.id2word.get(id, self.id2word[self.word2id['[UNK]']]) for id in ids]
            text = " ".join(words)
            # strip ['PAD]'] and ['[CLS]'] and ['[SEP]'] from the text
            if strip:
                text = text.replace('[PAD]', '')
                text = text.replace('[CLS]', '')
                text = text.replace('[SEP]', '')
            text = text.strip()
            texts.append(text)
        return texts

    def encode(self,
            text: str, 
            max_length: int = None,
            add_cls: bool = False,
            add_sep: bool = False, 
            return_attention_mask =True,
            padding: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
            pad_to_multiple_of=None,
            padding_side=None,
            return_tensors=None) -> dict:
        """Encode a string into a list of character IDs."""
        if type(text) is str:
            text = [text]
        # assert that text is a list of strings
        if not isinstance(text, list) or not all(isinstance(t, str) for t in text):
            raise ValueError("Input text must be a list of strings.")
        encodes = []
        rows = [t.split() for t in text]  # split each string into words
        word_ids = []
        for words in rows:
            assert len(words) > 0, "Input text must contain at least one word."
            encoded = self._encode(
                words, 
                pad_to_length=self.max_word_length, 
                max_length=max_length, 
                add_cls=add_cls, 
                add_sep=add_sep
            )
            if self.word2id is not None:
                word_ids.append(self.encode_to_word_ids(
                    words,
                    pad_to_length=0,
                    max_length=max_length,
                    add_cls=add_cls,
                    add_sep=add_sep
                ))
            encodes.append(encoded)
        out = {
            'input_ids': encodes,
        }
        if self.word2id is not None:
            out['word_ids'] = word_ids
        if return_attention_mask:
            attention_mask = [[1] * len(ids) for ids in encodes]
            out['attention_mask'] = attention_mask
        
        if padding is not None and padding != PaddingStrategy.DO_NOT_PAD:
            out = self.pad(
                encoded_inputs=out,
                padding=padding,
                max_length=max_length,
                pad_to_multiple_of=pad_to_multiple_of,
                padding_side=padding_side,
                return_attention_mask=return_attention_mask,
                return_tensors=return_tensors,
                verbose=False,
            )
        if return_tensors == "np" or return_tensors == "pt":
            for key in out.keys():
                out[key] = np.array(out[key])
                
                if return_tensors == "pt":
                    out[key] = torch.tensor(out[key], dtype=torch.int64)
        return out
            
    def __call__(
        self,
        text: Union[str, List[str], None] = None,
        # text_pair:  Union[str, List[str], None] = None,
        # text_target:  Union[str, List[str], None] = None,
        # text_pair_target: Optional[
        #      Union[str, List[str], None]
        # ] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str] = False,
        # truncation: Union[bool, str] = None,
        max_length: Optional[int] = None,
        # stride: int = 0,
        # is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        padding_side: Optional[str] = None,
        return_tensors: Optional[Union[str]] = None,
        # return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        # return_overflowing_tokens: bool = False,
        # return_special_tokens_mask: bool = False,
        # return_offsets_mapping: bool = False,
        # return_length: bool = False,
        # verbose: bool = True,
        add_cls: bool = None,
        add_sep: bool = None,
        **kwargs,
    ) -> Any:
        return self.encode(
            text=text,
            max_length=max_length,
            return_attention_mask=return_attention_mask if return_attention_mask else False,
            add_cls=add_special_tokens if add_cls is None else add_cls,
            add_sep=add_special_tokens if add_sep is None else add_sep,
            return_tensors=return_tensors,
            padding=padding,
            pad_to_multiple_of=pad_to_multiple_of,
            padding_side=padding_side if padding_side is not None else "right",
            # truncation=truncation,

        )
    
    def pad(
        self,
        encoded_inputs: Union[
            BatchEncoding,
            List[BatchEncoding],
            Dict[str, EncodedInput],
            Dict[str, List[EncodedInput]],
            List[Dict[str, EncodedInput]],
        ],
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        padding_side: Optional[str] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        verbose: bool = True,
    ) -> BatchEncoding:
        """
        Pad encoded inputs (on left/right and up to predefined length or max length in the batch)

        Args:
            encoded_inputs:
                Dictionary of tokenized inputs (`List[int]`) or batch of tokenized inputs (`List[List[int]]`).
            max_length: maximum length of the returned list and optionally padding length (see below).
                Will truncate by taking into account the special tokens.
            padding_strategy: PaddingStrategy to use for padding.

                - PaddingStrategy.LONGEST Pad to the longest sequence in the batch
                - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)
                - PaddingStrategy.DO_NOT_PAD: Do not pad
                The tokenizer padding sides are defined in `padding_side` argument:

                    - 'left': pads on the left of the sequences
                    - 'right': pads on the right of the sequences
            pad_to_multiple_of: (optional) Integer if set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Core on NVIDIA hardware with compute capability
                `>= 7.5` (Volta).
            padding_side:
                The side on which the model should have padding applied. Should be selected between ['right', 'left'].
                Default value is picked from the class attribute of the same name.
            return_attention_mask:
                (optional) Set to False to avoid returning attention mask (default: set to model specifics)
        """
        if padding_side is None:
            padding_side = "right"

        if isinstance(encoded_inputs, list):
            if len(encoded_inputs) == 0:
                raise ValueError("encoded_inputs list is empty; cannot pad")
            if isinstance(encoded_inputs[0], dict):
                batched_inputs: Dict[str, List[Any]] = {}
                for example in encoded_inputs:
                    for key, value in example.items():
                        batched_inputs.setdefault(key, []).append(value)
                encoded_inputs = batched_inputs
            else:
                raise ValueError(
                    "encoded_inputs must be a dict or list of dicts when padding"
                )

        if not isinstance(encoded_inputs, dict):
            raise ValueError(
                "encoded_inputs must be a dict or list of dicts when padding"
            )

        if max_length is None:
            if 'input_ids' not in encoded_inputs:
                raise ValueError(
                    "input_ids must be provided to infer max_length when padding"
                )
            max_length = max(len(ids) for ids in encoded_inputs['input_ids'])

        if isinstance(padding, bool):
            if padding:
                padding = PaddingStrategy.MAX_LENGTH
            else:
                padding = PaddingStrategy.DO_NOT_PAD
        elif isinstance(padding, str):
            padding = PaddingStrategy(padding)
        elif not isinstance(padding, ExplicitEnum):
            raise ValueError(
                f"padding should be a boolean, a string or an instance of PaddingStrategy, but is {type(padding)}"
            )
        if padding == PaddingStrategy.DO_NOT_PAD:
            return encoded_inputs
        elif padding == PaddingStrategy.LONGEST:
            max_length = max(len(ids) for ids in encoded_inputs['input_ids'])
            # make it less than max_length and divisble by pad_to_multiple_of if specified
            if pad_to_multiple_of is None:
                pad_to_multiple_of = 4
            max_length = (max_length + pad_to_multiple_of - 1) // pad_to_multiple_of * pad_to_multiple_of
        elif padding == PaddingStrategy.MAX_LENGTH:
            if max_length is None:
                raise ValueError("max_length must be specified when padding to max length.")
        else:
            raise ValueError(f"Unknown padding strategy: {padding}")    
        for key in encoded_inputs.keys():
            if key == 'input_ids':
                padded_ids = []
                for ids in encoded_inputs[key]:
                    if len(ids) < max_length:
                        if padding_side == "right":
                            ids += [self._encode_word('[PAD]')] * (max_length - len(ids))
                        else:
                            ids = ([self._encode_word('[PAD]')] * (max_length - len(ids))) + ids
                    padded_ids.append(ids[:max_length])
                encoded_inputs[key] = padded_ids
            elif key == 'attention_mask':
                attention_mask = []
                for mask in encoded_inputs['attention_mask']:
                    if padding_side == "right":
                        mask = mask + [0] * (max_length - len(mask))
                    else:
                        mask = [0] * (max_length - len(mask)) + mask
                    attention_mask.append(mask)
                encoded_inputs['attention_mask'] = attention_mask
            elif key == 'word_ids':
                padded_word_ids = []
                for word_ids in encoded_inputs[key]:
                    if len(word_ids) < max_length:
                        if padding_side == "right":
                            word_ids += [self.word2id['[PAD]']] * (max_length - len(word_ids))
                        else:
                            word_ids = ([self.word2id['[PAD]']] * (max_length - len(word_ids))) + word_ids
                    padded_word_ids.append(word_ids[:max_length])
                encoded_inputs[key] = padded_word_ids

        # Generate attention_mask if requested and not already present
        # attention_mask shape: (B, S) where S is seq_length (number of words)
        if return_attention_mask and 'attention_mask' not in encoded_inputs:
            attention_mask = []
            for ids in encoded_inputs['input_ids']:
                # ids has shape (S, C) where S is seq_length, C is num_characters
                # We create a mask of 1s for real tokens and 0s for padding
                seq_len = len(ids)
                mask = []
                pad_token = self._encode_word('[PAD]')
                for word_chars in ids:
                    # Check if this word is a padding token
                    if word_chars == pad_token:
                        mask.append(0)
                    else:
                        mask.append(1)
                attention_mask.append(mask)
            encoded_inputs['attention_mask'] = attention_mask

        if return_tensors in {"pt", "np"}:
            tensorizer = torch.tensor if return_tensors == "pt" else np.array
            for key, value in encoded_inputs.items():
                if isinstance(value, list):
                    encoded_inputs[key] = tensorizer(value)

        return encoded_inputs

    def save_pretrained(self, save_directory: str):
        """
        Save the tokenizer to a directory.
        """
        if not save_directory:
            raise ValueError("save_directory must be specified.")
        charset = {
            'valid_char2id': self.valid_char2id,
            'valid_id2char': self.valid_id2char,
            'special_char2name': self.special_char2name,
            'special_name2char': self.special_name2char,
            'special_token2char': self.special_token2char,
            'special_token2word': self.special_token2word,
            'special_word2token': self.special_word2token,
        }
        with open(f"{save_directory}/charset.json", 'w', encoding='utf-8') as f:
            json.dump(charset, f, indent=4)
        if self.word2id:
            with open(f"{save_directory}/word2id.json", 'w', encoding='utf-8') as f:
                json.dump(self.word2id, f, indent=4)    



if __name__ == "__main__":
    text = [
        "the quick brown fox jumps over the lazy dog",
        "hello world",
        "this is a test sentence",
        "another example of a sentence with a long word that exceeds the maximum length, way long but should be truncated to fit within the limit",
    ]
    tokenizer = CharTokenizer("tokenizer/charset.json", max_word_length=12)
    print([tokenizer.valid_char2id[tokenizer.special_name2char['pad']], tokenizer.valid_char2id[tokenizer.special_name2char['wpad']]])
    encoded = tokenizer(text, max_length=20, 
                        return_attention_mask=True, 
                        add_cls=True, add_sep=True, 
                        return_tensors="np", 
                        padding="longest")
    print("Encoded Input IDs:", encoded['input_ids'].shape)
    print("Attention Mask:", encoded['attention_mask'].shape)

    encoded_list = encoded['input_ids'].tolist()
    for row in encoded_list:
        decoded = tokenizer._decode_words(row, strip=True)
        print("Decoded Words:", " ".join(decoded))


