# Documents

**Start here**
- [README](../README.md): every measured result, the three commands, the disclaimer.
- [registry/models.json](../registry/models.json): the models with a fitted codebook, their file hashes, the rows measured on them.
- [reproduce-llamacpp.sh](../reproduce-llamacpp.sh): build the patched llama.cpp and re-run an audit from nothing.

**The live path in llama.cpp**
- [mscc/LLAMACPP-CPCA-DESIGN.md](../mscc/LLAMACPP-CPCA-DESIGN.md): why the change is small, the algebra, the milestone log, speed as measured.
- [mscc/llamacpp/README.md](../mscc/llamacpp/README.md): the patch series and how to apply it.
- [mscc/capture_gguf_kv.py](../mscc/capture_gguf_kv.py), [mscc/export_cpca_gguf.py](../mscc/export_cpca_gguf.py): fit a codebook for any GGUF; write it in the orientation llama.cpp reads.
- [tools/kvcache.py](../tools/kvcache.py): pull, serve, audit.

**The codec**
- [mscc/README.md](../mscc/README.md): frames at rest, the guard, the health stamp, rates.
- [mscc/CODE-SPACE-ATTENTION.md](../mscc/CODE-SPACE-ATTENTION.md): attention over the codes in PyTorch, the withdrawn RoPE claim, the cost table.
- [lib/codespace.py](../lib/codespace.py), [lib/codespace_selftest.py](../lib/codespace_selftest.py): the reference implementation and its CPU self-test.

**The benchmark**
- [auditor/README.md](../auditor/README.md): what it measures and how to run it.
- [auditor/SPEC-v0.1.md](../auditor/SPEC-v0.1.md): the protocol, the reference-arm rule, exclusion, the empty-reply retry.
- [auditor/FIRST-RESULT.md](../auditor/FIRST-RESULT.md): int4 is safe at 8B and harmful at 1.7B, and the five harness bugs found by testing.
- [auditor/PRIVACY.md](../auditor/PRIVACY.md): what a submission contains and what it never does.
- [results/](../results/): every result file behind a table in the README.
- [submissions/](../submissions/): the filed, validated runs the results page is built from.
