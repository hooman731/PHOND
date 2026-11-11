# PHOND

Code repository associated with the **PHOND** research project on Persian hate speech and offensive language detection.

## Repository Contents

| File | Content | Paper |
|---|---|---|
| `Tag_Analysis.ipynb` | Data tagging and analysis: the labeling pipeline, the 220-example reference corpus and the annotation prompt | Sections 4.1 to 4.3, Appendix A |
| `prompting_ablation.py` | Effect of the examples in the labeling prompt | Section 4.5 |
| `CommunityAnalysis.ipynb` | Community and user-network analysis | Section 5, Table 3 |
| `homophily.py` | Homophily under activity, degree and sampling controls | Table 4, Section 5.3 |
| `homophily_pps.py` | Homophily with user labels from the PPS texts only | Section 5.3 |
| `Experiments_P1.py` | Phase 1 experiments: text and network models | Table 5 |
| `network_ablation.py` | Ablation of the network terms, rewired graphs, graph without edges between test users, per class results | Table 6, Section 6.2, Appendix C |
| `finetune_parsbert.py` | Fine-tuned ParsBERT with the network features | Sections 6.1 and 6.2 |
| `finetune_bertweetfa.py` | Fine-tuned BERTweet-FA with the network features | Sections 6.1 and 6.2 |
| `Experiment_P2_LLM.ipynb` | Phase 2 LLM-based experiments | Table 7, Appendix B |

## Data

The PHOND dataset and user-network data are not included in this repository.

Please refer to the **Data Availability Statement** in the associated paper for information about data access. The data are deposited on Zenodo with controlled access: https://doi.org/10.5281/zenodo.22032230

The scripts read the dataset file (columns `Id`, `User_SId`, `Text`, `classification`, `hate_class`) and the edge file (two user id columns, such as `source` and `target`).

## Usage

Install the requirements with `pip install -r requirements.txt`. The fine-tuning scripts need a GPU.

Update the required dataset paths in the corresponding scripts or notebooks before running the experiments.

For Phase 1:

```bash
python Experiments_P1.py
```

Phase 2 experiments are provided in:

```text
Experiment_P2_LLM.ipynb
```

For Table 7, set the model name in `query_gpt` to one of the four models, as named on OpenRouter: `google/gemini-2.5-flash-lite`, `google/gemma-3n-e4b-it`, `qwen/qwen3-vl-8b-instruct` and `openai/gpt-5-mini`.

The other experiments:

```bash
python prompting_ablation.py --eval sample.csv
python homophily.py --csv phond.csv --edges edges.csv
python homophily_pps.py --csv phond.csv --edges edges.csv --pps pps.csv
python network_ablation.py --csv phond.csv --edges edges.csv
python finetune_parsbert.py --csv phond.csv --edges edges.csv
python finetune_bertweetfa.py --csv phond.csv --edges edges.csv
```

## Code Availability

The source code associated with the PHOND study is publicly available in this repository.

The version associated with the paper is permanently archived on Zenodo:

**DOI:** https://doi.org/10.5281/zenodo.22642668

## License

This code is released under the MIT License. See `LICENSE`.

## Dataset Citation

If you use the **PHOND dataset** in your research, please cite the associated PHOND paper once it is published.

Citation information will be added to this repository after publication.

## Citation

If you use the code or methodology provided in this repository, please cite the associated PHOND paper once it is published.

Full citation information will be added after publication.
