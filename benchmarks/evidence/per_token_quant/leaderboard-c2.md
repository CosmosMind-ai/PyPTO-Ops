# Leaderboard — campaign c2, decision level, every hard gate holds; target: latency_us <= 35.0 with every hard gate (correct: correct >= 1)

| # | eid | round | family | gain % | latency_us | correct | n_shapes | target | dir |
|---|---|---|---|---|---|---|---|---|---|
| 1 | e000041 | 7 | bf16_fused_spmd_r16_q256_with_fp32_stream | 73.89 | 27.56 | 1 | 1 | met | `designs/e000041_bf16_fused_spmd_r16_q256_with_fp32_stream` |
| 2 | e000040 | 6 | exact_hash_r16_q256_control | 66.16 | 35.72 | 1 | 1 | - | `designs/e000040_exact_hash_r16_q256_control` |
| 3 | e000035 | 5 | bf16_large_batch_streamed_factored_v2 | 64.47 | 37.5 | 1 | 1 | - | `designs/e000035_bf16_large_batch_streamed_factored_v2` |
| 4 | e000020 | 3 | bf16_chunk_promotion_matched_control | 63.41 | 38.62 | 1 | 1 | - | `designs/e000020_bf16_chunk_promotion_matched_control` |
| 5 | e000015 | None | resident_chunked_absmax | 63.18 | 38.86 | 1 | 1 | - | `designs/e000015_resident_chunked_absmax` |
| 6 | e000038 | 5 | bf16_large_batch_streamed_factored_v3 | 61.34 | 40.8 | 1 | 1 | - | `designs/e000038_bf16_large_batch_streamed_factored_v3` |
| 7 | e000028 | 4 | bf16_chunk_promotion_matched_control_v2 | 51.68 | 51 | 1 | 1 | - | `designs/e000028_bf16_chunk_promotion_matched_control_v2` |
| 8 | e000032 | 4 | bf16_chunk_promotion_matched_control_v3 | 49.29 | 53.52 | 1 | 1 | - | `designs/e000032_bf16_chunk_promotion_matched_control_v3` |
| 9 | e000014 | None | reference | - | 105.5 | 1 | 1 | - | `designs/e000014_reference` |
