# Training Recipes — Paper Table 2

Every specialist in the paper's Table 2 uses **one shared SFT recipe**:
`training/train_sft.py`. Only the dataset and adapter name change between
runs. This is the design point the paper emphasises — gains come from
dataset/task selection, not from per-benchmark hyperparameter tuning.

## The shared recipe

| Setting               | Value                                              |
|-----------------------|----------------------------------------------------|
| Base model            | `Qwen/Qwen2.5-7B-Instruct`                         |
| LoRA rank / alpha     | r=32, α=64                                         |
| LoRA target modules   | q,k,v,o,gate,up,down                               |
| Dropout               | 0.05                                               |
| Epochs                | 3                                                  |
| Batch size            | 8 per device × 2 grad accum (effective 16)         |
| Max sequence length   | 1024 (override per dataset)                        |
| Learning rate         | 2e-4, cosine schedule, 5 % warmup                  |
| Weight decay          | 0.0                                                |
| Seed                  | 42                                                 |
| Precision             | bf16                                               |
| Hardware tested       | single NVIDIA L40S (48 GB)                         |

Install training deps with `pip install -e ".[training]"`.

## Paper Table 2 — benchmark mapping

The strict-scorer gains below are from the paper, all under the shared
recipe above. "Adapter" is the HF id under
`pavan01729/adaptive-minds-loras/qwen2.5-7b/`.

| Benchmark                  | n   | Base   | Specialist gain | Adapter                       | Dataset notes                       |
|----------------------------|-----|--------|-----------------|-------------------------------|-------------------------------------|
| Spider-dev (SQL)           | 300 | 12.3 % | **+29.4 pp**    | `qwen25_sql_v1`               | Spider train split (NL→SQL)         |
| Text2Cypher                | 300 | 0.0 %  | **+39.3 pp**    | `qwen25_cypher_v1`            | Neo4j NL→Cypher (HF: `tomasonjo/text2cypher`) |
| nl2bash                    | 606 | 8.4 %  | **+8.9 pp**     | `qwen25_bash_v1`              | NL2Bash (HF: `xlangai/nl2bash`)     |
| LC-QuAD (SPARQL)           | 100 | 0.0 %  | **+59.0 pp**    | `qwen25_sparql_v1`            | LC-QuAD 2.0 NL→SPARQL               |
| MermaidSeqBench            | 100 | 4.5 %  | **+16.7 pp**    | `qwen25_mermaid_v1`           | Internal Mermaid sequence diagrams  |
| PII (NER+redaction)        | 100 | 2.7 %  | **+76.3 pp**    | `qwen25_pii_v1`               | Synthetic PII spans                 |
| Qiskit-HumanEval-hard      | 151 | 10.6 % | **+4.6 pp**     | `qwen25_quantum_v1_grpo`      | SFT warm + GRPO (execution reward)  |
| LEDGAR (legal, 3-stage)    | 100 | 0.0 %  | **+84.0 pp**    | `qwen25_legal_v1_grpo`        | 3-stage curriculum SFT → GRPO       |
| ChEMBL canonical SMILES    | 300 | 6.3 %  | **+30.4 pp**    | `qwen25_chem_v1_grpo`         | SFT warm + GRPO (QED reward)        |

Strong-base reasoning rows (no capability gap):

| Benchmark   | Base   | Specialist | Δ      | Adapter                       |
|-------------|--------|------------|--------|-------------------------------|
| MATH-500    | 69.2 % | 67.2 %     | −2.0   | `qwen25_reasoning_v1`         |
| GSM8K-250   | 86.0 % | 88.0 %     | +2.0   | `qwen25_reasoning_v1`         |

These two are documented in the paper as "specialists help where the base
has a real capability gap" — when the base is already strong the recipe
moves accuracy by less than ±2 pp (within seed noise).

## Running the shared recipe

```bash
python training/train_sft.py \
    --dataset hf://your-org/your-dataset \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --lora-name qwen25_my_specialist_v1 \
    --output-dir ./loras
```

Expected dataset schema: rows with `prompt` and `completion` columns.
The script joins them as `{prompt} {completion}` and trains causally on
the joined string.

### Datasets — where to get them

Most benchmarks use publicly available datasets:

- **Spider**: <https://yale-lily.github.io/spider>
- **Text2Cypher**: <https://huggingface.co/datasets/tomasonjo/text2cypher>
- **nl2bash**: <https://huggingface.co/datasets/xlangai/nl2bash>
- **LC-QuAD 2.0**: <https://figshare.com/projects/LCQuAD_2_0/62270>
- **LEDGAR**: <https://huggingface.co/datasets/lex_glue> (subset `ledgar`)
- **ChEMBL**: <https://www.ebi.ac.uk/chembl/>
- **Qiskit-HumanEval**: <https://github.com/qiskit-community/qiskit-humaneval>

PII, MermaidSeqBench, and the in-house validators used for the strict
scorer are described in the paper's appendices.

## GRPO + 3-stage curriculum (legal, chemistry, quantum)

These three specialists use the SFT recipe above as a warm-start
(stage 1), then GRPO with an execution-based reward (stage 2 / 3). The
GRPO step is not yet included in v0.1 — the published adapters above are
the final-stage checkpoints. The full GRPO recipe is on the roadmap (see
README §Roadmap).

## Notes

- The recipe is intentionally not tuned per benchmark. The paper's claim
  is that the same recipe produces meaningful gains across nine very
  different tasks — that uniformity is the result, not a missing detail.
- For datasets larger than ~50 K rows you may want to bump
  `--max-seq-length`. The default 1024 was chosen for the median
  Table-2 dataset.
- The adapters published on the Hub were trained on internal GPUs
  (single L40S). On smaller cards reduce `per_device_train_batch_size` to
  2–4 and proportionally bump `gradient_accumulation_steps` to keep the
  effective batch at 16.
