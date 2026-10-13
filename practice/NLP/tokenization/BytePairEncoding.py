class BPETokenizer:

    """
        Byte Pair Encoding (BPE) tokenizer for train and inference data for a language model.

        Three things to keep apart:
          - Bytes:  the text encoded as UTF-8, numbers 0-255. Any text in any
                    language can be written with them.
          - Tokens: IDs in the vocabulary. IDs 0-255 are the single bytes;
                    every ID from 256 up is a merge of two earlier tokens and
                    stands for a longer sequence of bytes.
          - Vocab:  the lookup from token ID -> the bytes it stands for.

        Training (fit) repeatedly finds the most frequent pair of neighbouring
        tokens and replaces it with a new token. The ordered list of merges is
        everything the tokenizer learns; the vocab is rebuilt from it.
    """

    N_BYTES = 256  # Size of the base vocabulary: one token per possible byte

    def __init__(self):
        self.vocab_size = None
        self.vocab = None     # {token_id: bytes}, built from the merges after fit()
        self.tokens = None    # The training text as token IDs (shrinks with each merge)
        self.merges = {}      # {(token_a, token_b): new_token_id}, in the order learned

    def fit(self,train_data,vocab_size=500):
        """
            Function to fit BPE tokenizer on the training data to create vocab.
            Learns (vocab_size - 256) merges on top of the 256 byte tokens.
        """
        merge_size = vocab_size - self.N_BYTES
        self.tokens = self._text_to_bytes(train_data)
        for merge in range(merge_size):
            # Recount every round: counts from before the last merge are stale
            pair_freq = self._count_pairs(self.tokens)
            if not pair_freq:
                break  # Everything is already merged into a single token
            pair = max(pair_freq,key=pair_freq.get)
            idx = merge + self.N_BYTES
            print(f"merging common pair {pair} as a new token {idx}")
            self.tokens = self._merge_list(self.tokens,pair,idx)
            self.merges[pair] = idx
        self._build_vocab()

    def encode(self,text) -> list[int]:
        """
            Converts text to token IDs by replaying the learned merges.

            Merges must be applied in the order they were learned, because later
            merges are built from earlier ones (e.g. 257 = (256, 97) only exists
            once 256 has been formed). So at each step we merge the pair that was
            learned earliest, i.e. has the smallest new token ID.
        """
        if self.vocab is None:
            raise RuntimeError("Call fit() before encode()")
        tokens = self._text_to_bytes(text)
        while len(tokens) >= 2:
            # Of the neighbouring pairs in the text, find the earliest learned merge
            pairs = set(zip(tokens,tokens[1:]))
            pair = min(pairs,key=lambda p: self.merges.get(p,float("inf")))
            if pair not in self.merges:
                break  # No pair in the text was ever merged during training
            tokens = self._merge_list(tokens,pair,self.merges[pair])
        return tokens

    def decode(self,tokens) -> str:
        """
            Converts token IDs back to text: look up each token's bytes, join
            them, then decode the bytes as UTF-8.
        """
        if self.vocab is None:
            raise RuntimeError("Call fit() before decode()")
        unknown = sorted(set(tokens) - self.vocab.keys())
        if unknown:
            raise ValueError(f"Token IDs not in the vocabulary: {unknown}")
        text_bytes = b"".join(self.vocab[t] for t in tokens)
        # A token sequence can end in the middle of a multi-byte character
        # (e.g. half an emoji); "replace" shows that as � instead of crashing
        return text_bytes.decode("utf-8",errors="replace")

    def _build_vocab(self):
        """
            Rebuilds the vocabulary {token_id: bytes} from the merges.
            Base tokens 0-255 are single bytes; each merged token is the bytes
            of its two parts joined. Dicts keep insertion order, so both parts
            of a merge are always in the vocab before the merge itself.
        """
        self.vocab = {i: bytes([i]) for i in range(self.N_BYTES)}
        for (a, b), idx in self.merges.items():
            self.vocab[idx] = self.vocab[a] + self.vocab[b]
        self.vocab_size = len(self.vocab)

    def _text_to_bytes(self,text):
        """
            Encodes text as UTF-8 and returns the bytes as a list of ints (0-255).
            These are the starting tokens before any merges.
        """
        return list(text.encode("utf-8"))

    def _count_pairs(self,tokens):
        """
            Counts how often each pair of neighbouring tokens occurs.
        """
        pair_freq = {}
        for pair in zip(tokens,tokens[1:]):
            pair_freq[pair] = pair_freq.get(pair,0) + 1
        return pair_freq

    def _merge_list(self,tokens,pair,idx):
        """
            Returns a new list where every occurrence of `pair` is replaced by `idx`.
        """
        # Build a new list instead of deleting in place: deleting shrinks the
        # list while the loop still runs to the original length
        merged = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
                merged.append(idx)
                i += 2  # skip both halves of the pair
            else:
                merged.append(tokens[i])
                i += 1
        return merged


if __name__ == "__main__":
    train_text = """
    Desert elephants or desert-adapted elephants are not a distinct species of elephant but are African bush elephants (Loxodonta africana) that have made their homes in the Namib and Sahara deserts in Africa. Previously they were classified as a subspecies of the African bush elephant, but this is no longer the case. Desert-dwelling elephants were once more widespread in Africa than they are at present; they are currently found only in Namibia which covers about 115,154 square kilometres (44,461 sq mi).[1] Elephants have traditionally lived in this area and in the earlier part of the 20th century there were about 3,000 in the Kunene Region. By the 1980s these had greatly diminished in number, however since then, conservation measures have been put in place and by 2013 the number of elephants had increased to about 600.[2] In 1995–1996 there were good rains in Namibia and the elephants expanded their range southwards to the Ugab River.[1]

    The desert elephants were absent from the southern Kunene Region during the war for independence. They moved north for safety, returning to the Ugab River in the mid 1990s by which time many indigenous people had moved into the area following Namibia's independence. Many of these new residents had no experience of living with wild elephants.[1]

    Desert-adapted baby elephants cuddling in Damaraland, Namibia
    Desert-adapted baby elephants cuddling in Damaraland, Namibia
    In the Hoanib River area male elephants have tusks, but about a third of the female elephants there are without tusks. Adult bull desert elephants are usually solitary and roam over large areas. One was recorded as travelling between the Skeleton Coast National Park and the Etosha National Park within the span of a few months. Other bulls have occasionally moved into the area from better-watered regions to the east. The family groups in which most desert elephants move are small and usually consist of a female elephant and her offspring or two sisters and their dependent young. They tend to stay near the ephemeral rivers where there is greater availability of food. Some groups are resident in the Hoarusib River valley and a single group stays permanently near the Hoanib River, while other groups move between the two rivers, a distance of about 70 kilometres (43 mi). They usually make the trek in a single night, when the temperature is cooler than by day. At certain times during the year they move inland along narrow traditional paths to mountain areas in search of myrrh bushes (Commiphora spp.) which seem to be a favourite source of food for them.[3]
        """

    tokenizer = BPETokenizer()
    tokenizer.fit(train_text,vocab_size=500)

    print("#######################################")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print("First 5 learned tokens:")
    for idx in range(tokenizer.N_BYTES, tokenizer.N_BYTES + 5):
        print(f"  {idx} -> {tokenizer.vocab[idx]}")

    # Compression: how many tokens the training text needs vs. its raw bytes
    n_bytes = len(train_text.encode("utf-8"))
    n_tokens = len(tokenizer.encode(train_text))
    print(f"Training text: {n_bytes} bytes -> {n_tokens} tokens ({n_bytes / n_tokens:.2f}x compression)")

    # New text, not seen in training: still encodable, because every
    # character falls back to its bytes
    text = "Unicode is not that scary! 😄"
    ids = tokenizer.encode(text)
    print(f"input text encoded: {ids}")
    print(f"input text as tokens: {[tokenizer.vocab[t] for t in ids]}")
    print(f"input text decoded: {tokenizer.decode(ids)}")
    print(f"Round trip OK: {tokenizer.decode(ids) == text}")
    print("#######################################")
