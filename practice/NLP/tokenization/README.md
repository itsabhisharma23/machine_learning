# Tokenization

A language model can't read text, only numbers. A **tokenizer** turns text into a list of integer IDs (tokens) and turns those IDs back into text. This folder has two tokenizers written from scratch:

| File | Tokenizer | One token is… |
|---|---|---|
| [CharLevelTokenization.py](CharLevelTokenization.py) | `CharLevelTokenizer` | one character |
| [BytePairEncoding.py](BytePairEncoding.py) | `BPETokenizer` | a byte, or a frequently seen sequence of bytes |

Both follow the same pattern: `fit()` learns a vocabulary from training text, `encode()` turns text into IDs, and `decode()` turns IDs back into text.

## Characters, bytes and tokens

These three words come up throughout, and they mean different things:

- **Character:** a Unicode code point, which is what Python gives you when you loop over a string. `"a"`, `"न"` and `"😄"` are one character each.
- **Byte:** a number from 0 to 255. UTF-8 stores every character as 1 to 4 bytes: `"a"` is the single byte `97`, while `"😄"` is the 4 bytes `240 159 152 132`. These 256 values are enough to write any text in any language.
- **Token:** an ID in the tokenizer's vocabulary. What a token stands for depends on the tokenizer: a character in the char-level tokenizer, and a byte or a sequence of bytes in BPE.

---

## Character-level tokenizer

The simplest possible tokenizer: every unique character in the training text becomes one token.

### How it works

1. **Build the vocabulary.** Collect every unique character in the training corpus and sort them. The sorted position of each character becomes its token ID.
2. **Encode.** Replace each character in a string with its ID, using the `stoi` (string → integer) table.
3. **Decode.** Replace each ID with its character, using the `itos` (integer → string) table, and join the result.

For the corpus `"My name is abhishek sharma. I love deep learning."`, the vocabulary has 21 characters:

```
[' ', '.', 'I', 'M', 'a', 'b', 'd', 'e', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 'v', 'y']
```

so `' '` is ID 0, `'.'` is ID 1, `'I'` is ID 2, and so on. Uppercase letters sort before lowercase ones because sorting follows Unicode code point order.

### Usage

```python
from CharLevelTokenization import CharLevelTokenizer

tokenizer = CharLevelTokenizer()
tokenizer.fit("My name is abhishek sharma. I love deep learning.")  # builds the vocabulary

ids = tokenizer.encode("I love India")
print(ids)                    # [2, 0, 12, 15, 19, 7, 0, 2, 14, 6, 10, 4]
print(tokenizer.decode(ids))  # I love India
print(tokenizer.vocab_size)   # 21
```

Run the demo with `python3 CharLevelTokenization.py`.

### API

| Member | Description |
|---|---|
| `fit(corpus)` | Builds the vocabulary and the encode/decode tables from `corpus`. Call this first. |
| `encode(text) -> list[int]` | Converts a string to a list of token IDs. |
| `decode(tokens) -> str` | Converts a list of token IDs back to a string. |
| `vocab` | Sorted list of unique characters. |
| `vocab_size` | Number of unique characters. |
| `encodings` | The `stoi` dict: character → ID. |
| `decodings` | The `itos` dict: ID → character. |

### Errors

| Situation | Error |
|---|---|
| `encode` or `decode` called before `fit` | `RuntimeError: Call fit() before encode()` (or `decode()`) |
| Text contains characters that weren't in the corpus | `ValueError: Characters not in the vocabulary: ['x', 'z']` |
| Token IDs outside the vocabulary | `ValueError: Token IDs not in the vocabulary: [99]` |

The `ValueError` lists every unknown character or ID at once. With the demo corpus, `encode("xyz")` reports `['x', 'z']`; `y` is fine because it appears in "My".

### Things to know

- **Only characters seen during `fit` can be encoded.** To handle any input, fit on a corpus that covers every character you expect, or extend the tokenizer with an `<unk>` token that unknown characters map to.
- **"Character" means Unicode code point, not visible letter.** Text in any language round-trips correctly, but one visible letter can be several tokens: the Hindi word `नमस्ते` is 6 tokens, because vowel signs and the virama are separate code points. A skin-toned emoji like 👋🏽 is 2 tokens.
- **The vocabulary depends on the corpus.** Calling `fit` with different text gives different IDs for the same character, so always encode and decode with the same tokenizer instance.

---

## Byte Pair Encoding (BPE) tokenizer

The approach used by GPT-2, GPT-4 and Llama 3. It starts from bytes, so it can encode *any* text, and then learns to merge byte sequences that appear often into single tokens, so common text needs fewer tokens.

### How it works

**Training (`fit`):**

1. Convert the training text to UTF-8 bytes. The base vocabulary is the 256 possible byte values, tokens 0–255.
2. Count every pair of neighbouring tokens.
3. Take the most frequent pair and give it a new token ID, starting at 256.
4. Replace every occurrence of that pair in the text with the new token.
5. Repeat steps 2–4 until the vocabulary reaches `vocab_size`, so `vocab_size - 256` merges.

The ordered list of merges (`self.merges`) is everything the tokenizer learns. The vocabulary is rebuilt from it: each new token's bytes are the bytes of its two parts joined together.

**Encoding:** convert the text to bytes, then replay the merges in the order they were learned. Order matters because later merges are built from earlier ones: a token made from `(256, 97)` can only form after token 256 exists.

**Decoding:** look up the bytes for each token, join them, and decode them as UTF-8.

### Walkthrough

Training on `"aaabdaaabac"` with `vocab_size=260` learns 4 merges:

| Round | Most frequent pair | New token | Bytes it stands for | Text after merging |
|---|---|---|---|---|
| start | | | | `a a a b d a a a b a c` (11 tokens) |
| 1 | (97, 97) = `a a` | 256 | `aa` | `256 a b d 256 a b a c` |
| 2 | (256, 97) = `aa a` | 257 | `aaa` | `257 b d 257 b a c` |
| 3 | (257, 98) = `aaa b` | 258 | `aaab` | `258 d 258 a c` |
| 4 | (258, 100) = `aaab d` | 259 | `aaabd` | `259 258 a c` (4 tokens) |

The 11 bytes compress to 4 tokens. New text still works: `encode("xaaab")` gives `[120, 258]`. The learned token 258 covers `aaab`, and `x`, which wasn't in training, falls back to its byte, 120.

### Usage

```python
from BytePairEncoding import BPETokenizer

tokenizer = BPETokenizer()
tokenizer.fit("aaabdaaabac", vocab_size=260)

print(tokenizer.merges)                 # {(97, 97): 256, (256, 97): 257, (257, 98): 258, (258, 100): 259}
print(tokenizer.vocab[259])             # b'aaabd'

ids = tokenizer.encode("aaabdaaabac")
print(ids)                              # [259, 258, 97, 99]
print(tokenizer.decode(ids))            # aaabdaaabac
```

Run the demo with `python3 BytePairEncoding.py`. It trains a 500-token vocabulary on a paragraph of English and Unicode text, then prints the first learned tokens, the compression ratio, and how a new sentence gets split into tokens.

### API

| Member | Description |
|---|---|
| `fit(train_data, vocab_size=500)` | Learns `vocab_size - 256` merges from `train_data` and builds the vocabulary. Stops early if the whole text has merged into one token. |
| `encode(text) -> list[int]` | Converts a string to token IDs by replaying the merges in order. |
| `decode(tokens) -> str` | Converts token IDs back to a string. |
| `merges` | `{(token_a, token_b): new_token_id}`, in the order the merges were learned. |
| `vocab` | `{token_id: bytes}`, the bytes each token stands for. |
| `vocab_size` | Number of tokens in the vocabulary (256 + number of merges). |
| `tokens` | The training text as token IDs after the final merge. Useful for checking what `fit` did. |
| `N_BYTES` | 256, the size of the base byte vocabulary. |

### Errors

| Situation | Error |
|---|---|
| `encode` or `decode` called before `fit` | `RuntimeError: Call fit() before encode()` (or `decode()`) |
| Token IDs outside the vocabulary | `ValueError: Token IDs not in the vocabulary: [999]` |

There's no "unknown character" error: every character can be written as bytes, and every byte is in the vocabulary.

### Things to know

- **Decoding can produce `�`.** One character can be split across several tokens, as 😄 is when it decodes to 4 separate byte tokens. If you decode only part of a token list, you can end up halfway through a character. `decode` uses `errors="replace"`, so that shows as `�` instead of raising an error.
- **Merges can cross word boundaries.** This version merges over the whole text, so tokens like `b'e '` (the end of one word plus a space) appear. Production tokenizers first split text into words, numbers and punctuation with a regex and only merge inside each piece.
- **It's an educational implementation.** It has no special tokens (like `<|endoftext|>`), no way to save or load the learned merges, and it runs in pure Python, so training on more than a few MB of text is slow. To train a real language model, use Hugging Face `tokenizers` or SentencePiece, or an existing tokenizer such as `tiktoken`.

---

## Comparing the two

| | Character-level | BPE |
|---|---|---|
| Base unit | Unicode character | Byte (0–255) |
| Vocabulary size | Number of unique characters in the corpus (21 in the demo) | 256 + number of merges (you choose; real LLMs use 32k–200k) |
| Text it has never seen | Fails on any character not in the corpus | Always works: unfamiliar text falls back to single bytes |
| Sequence length | One token per character, so long | Common words and pieces are one token, so shorter |
| Training | One pass to collect characters | Many passes, one per merge |
| Used by | Small educational models, e.g. a char-level GPT on one book | GPT-2, GPT-4, Llama 3 and most modern LLMs |

A good way to learn both: start with the char-level tokenizer to see the basic encode/decode idea, then compare how BPE handles the same text. It needs far fewer tokens and never meets a character it can't encode.
