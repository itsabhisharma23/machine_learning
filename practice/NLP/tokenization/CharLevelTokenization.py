class CharLevelTokenizer:
    """
        Tokenize at each char level for any language.
    """

    def __init__(self):
        self.vocab_size = None
        self.vocab = None
        self.encodings = None
        self.decodings = None

    def fit(self, corpus):
        """
            Build the vocabulary and the encode/decode tables from the corpus
        """
        self._set_vocabulary(corpus)
        # encode the chars
        stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.encodings = stoi

        # decode the tokens
        itos = {i: ch for i, ch in enumerate(self.vocab)}
        self.decodings = itos

    def _set_vocabulary(self, corpus):
        """
            Function to set vocabulary
        """
        chars = sorted(set(corpus))
        self.vocab_size = len(chars)
        self.vocab = chars

    def encode(self, text) -> list[int]:
        if self.encodings is None:
            raise RuntimeError("Call fit() before encode()")
        unknown = sorted(set(text) - self.encodings.keys())
        if unknown:
            raise ValueError(f"Characters not in the vocabulary: {unknown}")
        return [self.encodings[ch] for ch in text]

    def decode(self, tokens) -> str:
        if self.decodings is None:
            raise RuntimeError("Call fit() before decode()")
        unknown = sorted(set(tokens) - self.decodings.keys())
        if unknown:
            raise ValueError(f"Token IDs not in the vocabulary: {unknown}")
        return "".join([self.decodings[i] for i in tokens])


if __name__ == "__main__":
    corpus = "My name is abhishek sharma. I love deep learning."

    tokenizer = CharLevelTokenizer()
    tokenizer.fit(corpus)

    text = "I love India"

    print("#######################################")
    print(f"input text encoded: {tokenizer.encode(text)}")
    print(f"input text decoded: {tokenizer.decode(tokenizer.encode(text))}")
    print("#######################################")

    print(f"Total Vocabulary Size: {tokenizer.vocab_size}")
    print(f"Total Vocabulary: {tokenizer.vocab}")

    
