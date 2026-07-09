"""Small plaintext smoke demo for the Pisces retrieval scaffold."""

import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from NssMPC.application.rag.pisces import PiscesConfig, PiscesRetriever


def main():
    torch.manual_seed(7)
    num_docs = 8
    hidden = 16
    vocab = 32
    doc_len = 6

    config = PiscesConfig(top_k=2, semantic_candidates=4, simhash_bits=32)
    retriever = PiscesRetriever(config)

    query_embedding = torch.randn(1, hidden)
    doc_embeddings = torch.randn(num_docs, hidden)
    doc_tokens = torch.randn(num_docs, doc_len, vocab)

    sem_scores, sem_ind, sem_docs = retriever.semantic_path(
        query_embedding,
        doc_embeddings,
        document_payload=doc_tokens,
    )

    term_frequency = torch.randint(0, 4, (vocab, num_docs)).float()
    query_tokens = torch.tensor([3, 5, 8])
    document_lengths = torch.randint(8, 24, (num_docs,)).float()
    lex_scores, lex_ind, lex_docs = retriever.lexical_path_plain(
        query_tokens,
        term_frequency,
        document_lengths,
        document_payload=doc_tokens,
    )

    print("semantic scores shape:", tuple(sem_scores.shape))
    print("semantic indicators shape:", tuple(sem_ind.shape))
    print("semantic docs shape:", tuple(sem_docs.shape))
    print("lexical scores shape:", tuple(lex_scores.shape))
    print("lexical indicators shape:", tuple(lex_ind.shape))
    print("lexical docs shape:", tuple(lex_docs.shape))


if __name__ == "__main__":
    main()
