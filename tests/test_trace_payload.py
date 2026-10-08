import numpy as np

from glassbox.trace_payload import build_step_payload


class TokenizerStub:
    def decode(self, token_id: int) -> str:
        return f"token-{token_id}"


def test_payload_reports_full_vocabulary_entropy_and_decodes_top_200():
    logits = np.linspace(-4, 4, 256, dtype=np.float32)
    shifted = logits.astype(np.float64)
    shifted -= shifted.max()
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    nonzero = probabilities[probabilities > 0]
    expected_entropy = round(float(-(nonzero * np.log2(nonzero)).sum()), 3)

    payload = build_step_payload(
        step_idx=0,
        token_id=255,
        token_str=" token",
        tokens_per_sec=1.0,
        cache_active=True,
        logits=logits,
        traces=[],
        tokenizer=TokenizerStub(),
        model=None,
        top_n=200,
        include_attention=False,
        include_lens=False,
    )

    assert payload["entropy_bits"] == expected_entropy
    assert len(payload["top_logit_strs"]) == 200