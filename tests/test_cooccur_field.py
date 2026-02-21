"""Tests for CooccurField — corpus-based bigram/trigram frequency model."""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from molequla import CooccurField, EvolvingTokenizer, corpus_generate


def _make_field():
    docs = [
        "Hello world hello world",
        "This is a test sentence",
        "Hello hello hello"
    ]
    tok = EvolvingTokenizer(docs)
    field = CooccurField()
    field.build_from_corpus(tok, docs)
    return field, tok, docs


class TestCooccurField(unittest.TestCase):

    def test_build_populates_unigram(self):
        """After building, unigram counts should be populated."""
        field, tok, docs = _make_field()
        self.assertGreater(field.total_tokens, 0)
        self.assertGreater(len(field.unigram), 0)

    def test_build_populates_bigram(self):
        """After building, bigram counts should exist."""
        field, tok, docs = _make_field()
        self.assertGreater(len(field.bigram), 0)

    def test_build_populates_trigram(self):
        """After building, trigram counts should exist."""
        field, tok, docs = _make_field()
        self.assertGreater(len(field.trigram), 0)

    def test_unigram_reflects_frequency(self):
        """Frequent characters should have higher unigram counts."""
        field, tok, docs = _make_field()
        # 'l' appears many times in "Hello hello hello"
        l_id = tok.stoi.get('l', -1)
        if l_id >= 0:
            self.assertGreater(field.unigram[l_id], 0)

    def test_sample_next_returns_valid_id(self):
        """sample_next should return a valid token ID."""
        field, tok, docs = _make_field()
        # Get some context IDs
        ids = tok.encode("Hello")
        if len(ids) > 1:
            next_id = field.sample_next(ids[1:])  # skip BOS
            self.assertGreaterEqual(next_id, 0)
            self.assertLess(next_id, tok.vocab_size)

    def test_sample_next_empty_context(self):
        """sample_next with empty context should fallback to unigram."""
        field, tok, docs = _make_field()
        next_id = field.sample_next([])
        self.assertGreaterEqual(next_id, 0)

    def test_sample_next_single_context(self):
        """sample_next with single-token context should use bigram."""
        field, tok, docs = _make_field()
        h_id = tok.stoi.get('H', -1)
        if h_id >= 0:
            next_id = field.sample_next([h_id])
            self.assertGreaterEqual(next_id, 0)

    def test_corpus_generate_returns_string(self):
        """corpus_generate should return a non-empty string."""
        field, tok, docs = _make_field()
        result = corpus_generate(tok, field, "Hello")
        self.assertIsInstance(result, str)

    def test_build_from_corpus_idempotent(self):
        """Rebuilding should produce consistent counts."""
        field, tok, docs = _make_field()
        count1 = field.total_tokens
        field.build_from_corpus(tok, docs)
        count2 = field.total_tokens
        self.assertEqual(count1, count2)

    def test_build_clears_previous(self):
        """Rebuilding should clear old data first."""
        field, tok, docs = _make_field()
        # Build with different docs
        new_docs = ["Completely different text"]
        field.build_from_corpus(tok, new_docs)
        # Should not contain counts from original docs
        # (total tokens should be different)
        self.assertLess(field.total_tokens, 100)


class TestFourgram(unittest.TestCase):
    """Tests for 4-gram support in CooccurField."""

    def test_build_populates_fourgram(self):
        field, tok, docs = _make_field()
        self.assertGreater(len(field.fourgram), 0)

    def test_fourgram_counts_match(self):
        docs = ["abcde"]
        tok = EvolvingTokenizer(docs)
        field = CooccurField()
        field.build_from_corpus(tok, docs)
        ids = tok.encode("abcde")
        if len(ids) >= 5:
            ctx = (ids[1], ids[2], ids[3])
            self.assertIn(ctx, field.fourgram)
            self.assertGreater(field.fourgram[ctx][ids[4]], 0)

    def test_ingest_adds_fourgram(self):
        field, tok, docs = _make_field()
        before = len(field.fourgram)
        field.ingest_tokens(tok.encode("zyxwvut"))
        after = len(field.fourgram)
        self.assertGreater(after, before)

    def test_rebuild_clears_fourgram(self):
        field, tok, docs = _make_field()
        field.ingest_tokens(tok.encode("unique sequence of tokens here"))
        before = len(field.fourgram)
        field.build_from_corpus(tok, ["ab"])
        after = len(field.fourgram)
        self.assertLess(after, before)


class TestCooccurWindow(unittest.TestCase):
    """Tests for co-occurrence window (Stanley-style proximity)."""

    def test_build_populates_cooccur_window(self):
        field, tok, docs = _make_field()
        self.assertGreater(len(field.cooccur_window), 0)

    def test_cooccur_window_symmetry(self):
        docs = ["ab"]
        tok = EvolvingTokenizer(docs)
        field = CooccurField()
        field.build_from_corpus(tok, docs)
        ids = tok.encode("ab")
        if len(ids) >= 3:
            a, b = ids[1], ids[2]
            self.assertIn(b, field.cooccur_window[a])
            self.assertIn(a, field.cooccur_window[b])

    def test_ingest_adds_cooccur(self):
        field, tok, docs = _make_field()
        total_before = sum(
            cnt for neighbors in field.cooccur_window.values()
            for cnt in neighbors.values()
        )
        field.ingest_tokens(tok.encode("new text here"))
        total_after = sum(
            cnt for neighbors in field.cooccur_window.values()
            for cnt in neighbors.values()
        )
        self.assertGreater(total_after, total_before)

    def test_rebuild_clears_cooccur_window(self):
        field, tok, docs = _make_field()
        before = len(field.cooccur_window)
        field.build_from_corpus(tok, ["a"])
        after = len(field.cooccur_window)
        self.assertLess(after, before)


class TestUserBoost(unittest.TestCase):
    """Tests for user word boost (Leo-style vocabulary absorption)."""

    def test_absorb_adds_boosts(self):
        field, tok, docs = _make_field()
        user_ids = tok.encode("Hello")
        field.absorb_user_words(user_ids)
        found = any(field.user_boost.get(i, 0) > 0 for i in user_ids)
        self.assertTrue(found)

    def test_absorb_decays_old(self):
        field, tok, docs = _make_field()
        first_ids = tok.encode("Hello")
        field.absorb_user_words(first_ids)
        old_boost = field.user_boost.get(first_ids[1], 0)
        second_ids = tok.encode("world")
        field.absorb_user_words(second_ids)
        new_boost = field.user_boost.get(first_ids[1], 0)
        self.assertLess(new_boost, old_boost)

    def test_decay_reduces(self):
        field, tok, docs = _make_field()
        field.absorb_user_words(tok.encode("test"))
        max_before = max(field.user_boost.values()) if field.user_boost else 0
        field.decay_user_boost()
        max_after = max(field.user_boost.values()) if field.user_boost else 0
        self.assertLess(max_after, max_before)

    def test_decay_removes_small(self):
        field = CooccurField()
        field.user_boost[42] = 0.005
        field.decay_user_boost()
        self.assertNotIn(42, field.user_boost)


class TestWeightedIngestion(unittest.TestCase):
    """Tests for resonance-weighted ingestion."""

    def test_high_weight_stronger_counts(self):
        docs = ["ab"]
        tok = EvolvingTokenizer(docs)
        ids = tok.encode("cd")

        field1 = CooccurField()
        field1.build_from_corpus(tok, docs)
        field1.ingest_tokens_weighted(ids, 1.0)

        field2 = CooccurField()
        field2.build_from_corpus(tok, docs)
        field2.ingest_tokens_weighted(ids, 5.0)

        if len(ids) >= 2:
            c1 = field1.bigram.get(ids[0], {}).get(ids[1], 0)
            c2 = field2.bigram.get(ids[0], {}).get(ids[1], 0)
            self.assertGreater(c2, c1)

    def test_zero_weight_no_change(self):
        field, tok, docs = _make_field()
        ids = tok.encode("Hello")
        before = dict(field.unigram)
        field.ingest_tokens_weighted(ids, 0.0)
        after = dict(field.unigram)
        for k in before:
            self.assertAlmostEqual(before[k], after.get(k, 0), places=6)

    def test_weighted_adds_fourgram(self):
        field, tok, docs = _make_field()
        before = len(field.fourgram)
        field.ingest_tokens_weighted(tok.encode("unique abcdefgh"), 2.0)
        after = len(field.fourgram)
        self.assertGreater(after, before)

    def test_weighted_adds_cooccur(self):
        field, tok, docs = _make_field()
        total_before = sum(
            cnt for neighbors in field.cooccur_window.values()
            for cnt in neighbors.values()
        )
        field.ingest_tokens_weighted(tok.encode("new text"), 1.5)
        total_after = sum(
            cnt for neighbors in field.cooccur_window.values()
            for cnt in neighbors.values()
        )
        self.assertGreater(total_after, total_before)


class TestSampleNextEnhanced(unittest.TestCase):
    """Tests for enhanced sample_next with 4-gram, cooccur, user boost."""

    def test_sample_with_user_boost(self):
        field, tok, docs = _make_field()
        ids = tok.encode("Hello")
        if len(ids) < 2:
            self.skipTest("Too few tokens")
        target = ids[1]
        field.user_boost[target] = 10.0
        hits = sum(1 for _ in range(200) if field.sample_next(ids[:1]) == target)
        self.assertGreater(hits, 10)

    def test_sample_returns_valid(self):
        field, tok, docs = _make_field()
        for _ in range(50):
            nxt = field.sample_next([])
            self.assertGreaterEqual(nxt, 0)
            self.assertLess(nxt, tok.vocab_size)


class TestFullPipelineEnhanced(unittest.TestCase):
    """Integration test: build → absorb → ingest weighted → sample → decay."""

    def test_pipeline(self):
        field, tok, docs = _make_field()
        self.assertGreater(field.total_tokens, 0)

        # Absorb user words
        user_ids = tok.encode("Hello world")
        field.absorb_user_words(user_ids)
        self.assertGreater(len(field.user_boost), 0)

        # Weighted ingest
        field.ingest_tokens_weighted(tok.encode("test words"), 1.5)

        # Sample
        ctx = tok.encode("Hello")
        if len(ctx) > 1:
            nxt = field.sample_next(ctx[1:])
            self.assertGreaterEqual(nxt, 0)
            self.assertLess(nxt, tok.vocab_size)

        # Decay
        max_before = max(field.user_boost.values())
        field.decay_user_boost()
        max_after = max(field.user_boost.values()) if field.user_boost else 0
        self.assertLess(max_after, max_before)

        # Structures populated
        self.assertGreater(len(field.fourgram), 0)
        self.assertGreater(len(field.cooccur_window), 0)


if __name__ == "__main__":
    unittest.main()
