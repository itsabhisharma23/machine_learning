# Character-Level Tokenizer

A minimal tokenizer that treats every character as one token. It is the simplest way to turn text into integers for a language model, and a good starting point before moving on to subword tokenizers like BPE.

## How it works

1. **Build the vocabulary.** Collect every unique character in a training corpus and sort them. The sorted position of each character becomes its token ID.
2. **Encode.** Replace each character in a string with its ID, using the `stoi` (string → integer) table.
3. **Decode.** Replace each ID with its character, using the `itos` (integer → string) table, and join the result.

For the corpus `"My name is abhishek sharma. I love deep learning."`, the vocabulary has 21 characters:

```
[' ', '.', 'I', 'M', 'a', 'b', 'd', 'e', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 'v', 'y']
```

so `' '` is ID 0, `'.'` is ID 1, `'I'` is ID 2, and so on. Uppercase letters sort before lowercase ones because sorting follows Unicode code point order.

## Usage

```python
from CharLevelTokenization import CharLevelTokenizer

tokenizer = CharLevelTokenizer()
tokenizer.fit("My name is abhishek sharma. I love deep learning.")  # builds the vocabulary

ids = tokenizer.encode("I love India")
print(ids)                    # [2, 0, 12, 15, 19, 7, 0, 2, 14, 6, 10, 4]
print(tokenizer.decode(ids))  # I love India
print(tokenizer.vocab_size)   # 21
```

Run the built-in demo with:

```bash
python3 CharLevelTokenization.py
```

## API

| Member | Description |
|---|---|
| `fit(corpus)` | Builds the vocabulary and the encode/decode tables from `corpus`. Call this first. |
| `encode(text) -> list[int]` | Converts a string to a list of token IDs. |
| `decode(tokens) -> str` | Converts a list of token IDs back to a string. |
| `vocab` | Sorted list of unique characters. |
| `vocab_size` | Number of unique characters. |
| `encodings` | The `stoi` dict: character → ID. |
| `decodings` | The `itos` dict: ID → character. |

## Errors

| Situation | Error |
|---|---|
| `encode` or `decode` called before `fit` | `RuntimeError: Call fit() before encode()` (or `decode()`) |
| Text contains characters that weren't in the corpus | `ValueError: Characters not in the vocabulary: ['x', 'z']` |
| Token IDs outside the vocabulary | `ValueError: Token IDs not in the vocabulary: [99]` |

The `ValueError` lists every unknown character or ID at once, so you can see everything that's missing from the corpus in one go. With the demo corpus, `encode("xyz")` reports `['x', 'z']`; `y` is fine because it appears in "My".

## Things to know

- **Only characters seen during `fit` can be encoded.** To handle any input, fit on a corpus that covers every character you expect, or extend the tokenizer with an `<unk>` token that unknown characters map to.
- **"Character" means Unicode code point.** Python iterates over strings by code point, so text in any language round-trips correctly. But one visible letter can be several tokens: the Hindi word `नमस्ते` is 6 tokens, because vowel signs and the virama are separate code points. A skin-toned emoji like 👋🏽 is 2 tokens.
- **The vocabulary depends on the corpus.** Calling `fit` with different text gives different IDs for the same character, so always encode and decode with the same tokenizer instance.

## Character-level vs. other tokenizers

| | Character-level | Word-level | Subword (BPE, WordPiece) |
|---|---|---|---|
| Vocabulary size | Tiny (tens to hundreds) | Huge (100k+) | Medium (30k–100k) |
| Unknown words | None, if every character is in the vocabulary | Common | Rare |
| Sequence length | Long: one token per character | Short | Medium |
| What each token means | Little on its own | A full word | A word piece |

Character-level tokenization keeps the vocabulary small and never meets an unknown *word*. The cost is much longer sequences, so the model has to learn spelling and word structure itself. That makes it a good fit for small educational models, such as a character-level GPT trained on a single book, while production LLMs use subword tokenizers.
