import os
import json
import argparse
import numpy as np
import pandas as pd
import yaml
from dataclasses import dataclass, asdict
from tqdm import tqdm

dataset_to_metrics = {
    "json_kv": "substring_exact_match",
    "nq": "substring_exact_match",
    "popqa": "substring_exact_match",
    "triviaqa": "substring_exact_match",
    "hotpotqa": "substring_exact_match",
    
    "narrativeqa": ["gpt-4-score"],
    "msmarco_rerank_psg": "NDCG@10",
    
    "trec_coarse": "exact_match",
    "trec_fine": "exact_match",
    "banking77": "exact_match",
    "clinic150": "exact_match",
    "nlu": "exact_match",
    
    "qmsum": "rougeL_recall",
    "multi_lexsum": ["gpt-4-f1"],
    
    "ruler_niah_s_1": "ruler_recall",
    "ruler_niah_s_2": "ruler_recall",
    "ruler_niah_s_3": "ruler_recall",
    "ruler_niah_mk_1": "ruler_recall",
    "ruler_niah_mk_2": "ruler_recall",
    "ruler_niah_mk_3": "ruler_recall",
    "ruler_niah_mq": "ruler_recall",
    "ruler_niah_mv": "ruler_recall",
    "ruler_fwe": "ruler_recall",
    "ruler_cwe": "ruler_recall",
    "ruler_vt": "ruler_recall",
    "ruler_qa_1": "substring_exact_match",
    "ruler_qa_2": "substring_exact_match",
    
    "infbench_qa": ["rougeL_f1"],
    "infbench_choice": ["exact_match"],
    "infbench_sum": ["gpt-4-f1"],
    
    "alce_asqa": ["str_em", "citation_rec", "citation_prec"],
    "alce_qampari": ["qampari_rec_top5", "citation_rec", "citation_prec"],
}

dataset_to_metrics = {k: [v] if isinstance(v, str) else v for k, v in dataset_to_metrics.items()}
custom_avgs = {
    "Recall": ["json_kv substring_exact_match", "ruler_niah_mk_2 ruler_recall", "ruler_niah_mk_3 ruler_recall", "ruler_niah_mv ruler_recall"],
    "RAG": ['nq substring_exact_match', 'hotpotqa substring_exact_match', 'popqa substring_exact_match', 'triviaqa substring_exact_match',],
    "ICL": ['trec_coarse exact_match', 'trec_fine exact_match', 'banking77 exact_match', 'clinic150 exact_match', 'nlu exact_match'],
    "Cite": ['alce_asqa str_em', 'alce_asqa citation_rec', 'alce_asqa citation_prec', 'alce_qampari qampari_rec_top5', 'alce_qampari citation_rec', 'alce_qampari citation_prec', ],
    "Re-rank": ['msmarco_rerank_psg NDCG@10', ],
    "LongQA": ['narrativeqa gpt-4-score', 'infbench_qa rougeL_f1', 'infbench_choice exact_match', ],
    "Summ": ['infbench_sum gpt-4-f1', 'multi_lexsum gpt-4-f1', ],
    # "RULER": ['ruler_niah_s_1 ruler_recall', 'ruler_niah_s_2 ruler_recall', 'ruler_niah_s_3 ruler_recall', 'ruler_niah_mk_1 ruler_recall', 'ruler_niah_mk_2 ruler_recall', 'ruler_niah_mk_3 ruler_recall', 'ruler_niah_mq ruler_recall', 'ruler_niah_mv ruler_recall', 'ruler_cwe ruler_recall', 'ruler_fwe ruler_recall', 'ruler_vt ruler_recall', 'ruler_qa_1 substring_exact_match', 'ruler_qa_2 substring_exact_match'],
    "Ours": ['Recall', 'RAG', 'ICL', 'Cite', 'Re-rank', 'LongQA', 'Summ'],
}

@dataclass
class arguments:
    tag: str = "v1"
    input_max_length: int = 131072
    generation_max_length: int = 100
    generation_min_length: int = 0
    max_test_samples: int = 100
    shots: int = 2
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    use_chat_template: bool = False
    seed: int = 42
    test_name: str = ""
    dataset: str = "nq"
    output_dir: str = "output"
    popularity_threshold: float = 3
        
    category: str = "synthetic"
    
    def update(self, new):
        for key, value in new.items():
            if hasattr(self, key):
                setattr(self, key, value)
                
    def get_path(self):
        tag = self.tag
        path = os.path.join(self.output_dir, "{args.dataset}_{tag}_{args.test_name}_in{args.input_max_length}_size{args.max_test_samples}_shots{args.shots}_samp{args.do_sample}max{args.generation_max_length}min{args.generation_min_length}t{args.temperature}p{args.top_p}_chat{args.use_chat_template}_{args.seed}.json".format(args=self, tag=tag))

        if os.path.exists(path.replace(".json", "-gpt4eval_o.json")):
            return path.replace(".json", "-gpt4eval_o.json")
        if "alce" in self.dataset:
            return path.replace(".json", ".json.score")
        
        if os.path.exists(path + ".score"):
            return path + ".score"
        return path

    def get_metric_name(self):
        for d, m in dataset_to_metrics.items():
            if d in self.dataset:
                return d, m
        return None
    
    def get_averaged_metric(self):
        path = self.get_path()
        print(path)
        if not os.path.exists(path):
            print("path doesn't exist")
            return None
        with open(path) as f:
            results = json.load(f)
        
        _, metric = self.get_metric_name()
        if path.endswith(".score"):
            if any([m not in results for m in metric]):
                print("metric doesn't exist")
                return None
            s = {m: results[m] for m in metric}
        else:
            if any([m not in results["averaged_metrics"] for m in metric]):
                print("metric doesn't exist")
                return None
            s = {m: results['averaged_metrics'][m] for m in metric}
        
        s = {m : v * (100 if m == "gpt-4-f1" else 1) * (100/3 if m == "gpt-4-score" else 1) for m, v in s.items()}
        print("found scores:", s)
        return s
        
    def get_metric_by_depth(self):
        path = self.get_path()
        path = path.replace(".score", '')
        print(path)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            results = json.load(f)

        output = []        
        _, metric = self.get_metric_name()
        metric = metric[0]
        keys = ["depth", "k", metric]
        for d in results["data"]:
            o = {}
            for key in keys:
                if key == "k" and "ctxs" in d:
                    d["k"] = len(d['ctxs'])
                if key not in d:
                    print("no", key)
                    return None
                o[key] = d[key]
            o["metric"] = o.pop(metric)
            output.append(o)
        
        df = pd.DataFrame(output)
        dfs = df.groupby(list(output[0].keys())[:-1]).mean().reset_index()

        return dfs.to_dict("records")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect HELMET results across datasets and emit a single CSV row per model.")
    parser.add_argument("--model", required=True, help="Model name (must match the output folder name under output_dir).")
    parser.add_argument("--training_length", type=int, required=True, help="Training length (context length) for the model.")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory containing this model's outputs. Defaults to output/<model>.",
    )
    parser.add_argument("--tag", default="v1", help="Tag used in output filenames (default: v1).")
    parser.add_argument(
        "--use_chat_template",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether this model was run with chat templates enabled (default: True).",
    )
    parser.add_argument(
        "--config_files",
        nargs="*",
        default=[
            "configs/recall.yaml",
            "configs/recall_short.yaml",
            "configs/rag.yaml",
            "configs/rag_short.yaml",
            "configs/longqa.yaml",
            "configs/longqa_short.yaml",
            "configs/summ.yaml",
            "configs/summ_short.yaml",
            "configs/rerank.yaml",
            "configs/rerank_short.yaml",
            "configs/icl.yaml",
            "configs/icl_short.yaml",
            "configs/cite.yaml",
            "configs/cite_short.yaml",
            "configs/ruler.yaml",
            "configs/ruler_short.yaml",
        ],
        help="Config YAMLs to load (defaults to the full suite).",
    )
    cli_args = parser.parse_args()

    # Single model config from CLI (kept as a list to preserve downstream logic)
    models_configs = [
        {
            "model": cli_args.model,
            "use_chat_template": cli_args.use_chat_template,
            "training_length": cli_args.training_length,
        }
    ]

    # check if training length is less than or equal to 65536 then use only configs which has "short" in the name
    if cli_args.training_length <= 65536:
        print("using short configs")
        cli_args.config_files = [file for file in cli_args.config_files if "short" in file]
    else:
        print("using long configs")
        cli_args.config_files = [file for file in cli_args.config_files if not "short" in file]


    # set your configs here, only include the ones that you ran
    config_files = cli_args.config_files

    dataset_configs = []
    for file in config_files:
        c = yaml.safe_load(open(file))
        
        if isinstance(c["generation_max_length"], int):
            c["generation_max_length"] = ",".join([str(c["generation_max_length"])] * len(c["datasets"].split(",")))
        for d, t, l, g in zip(c['datasets'].split(','), c['test_files'].split(','), c['input_max_length'].split(','), c['generation_max_length'].split(',')):
            dataset_configs.append({"dataset": d, "test_name": os.path.basename(os.path.splitext(t)[0]), "input_max_length": int(l), "generation_max_length": int(g), "max_test_samples": c['max_test_samples'], 'use_chat_template': c['use_chat_template'], 'shots': c['shots']})
    print(dataset_configs)    

    failed_paths = []
    df = []
    for model in tqdm(models_configs):
        args = arguments()
        args.tag = cli_args.tag
        args.output_dir = cli_args.output_dir or f"output/{model['model']}"
    
        for dataset in dataset_configs:
            args.update(model)
            args.update(dataset)

            metric = args.get_averaged_metric()
            dsimple, mnames = args.get_metric_name()

            if metric is None:
                failed_paths.append(args.get_path())
                continue
                
            for k, m in metric.items():
                df.append({**asdict(args), **model,
                    "metric name": k, "metric": m, 
                    "dataset_simple": dsimple + " " + k, "test_data": f"{args.dataset}-{args.test_name}-{args.input_max_length}"
                })

    all_df = pd.DataFrame(df)
    lf_df = all_df.pivot_table(index=["input_max_length", "model", ], columns="dataset_simple", values="metric", sort=False)
    lf_df = lf_df.reset_index()

    for k, v in custom_avgs.items():
        lf_df[k] = lf_df[v].mean(axis=1)

    print(lf_df.to_csv(index=False))

    results_csv_path = os.path.join(cli_args.output_dir or "output", f"{model['model']}_results.csv")
    lf_df.to_csv(results_csv_path, index=False)
    print(f"Results saved to {results_csv_path}")

    # print average for all columns for columns in custom_avgs['Ours]
    for col in custom_avgs['Ours']:
        print(f"Average for {col}: {lf_df[col].mean():.02f}")
    print(f"Average for \"Ours\": {lf_df['Ours'].mean():.02f}")
    
    print_paths = False
    if print_paths:
        print("Warning, failed to get the following paths, make sure that these are correct or the printed results will not be accurate:", failed_paths)
    # import pdb; pdb.set_trace()