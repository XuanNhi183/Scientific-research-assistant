import argparse
import yaml
from dataset_builder.dataset_builder import DatasetBuilder


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Build SFT fine-tuning dataset from arXiv papers.")
    parser.add_argument(
        "--config",
        default="config/dataset_config.yaml",
        help="Path to YAML config file (default: config/dataset_config.yaml)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds_cfg = cfg["dataset"]
    src_cfg = cfg["source"]

    print(f"Config loaded from: {args.config}")
    print(f"  output_path     : {ds_cfg['output_path']}")
    print(f"  samples_per_paper: {ds_cfg['samples_per_paper']}")
    print(f"  n_papers        : {src_cfg['n_papers']}")
    print(f"  categories      : {src_cfg['categories']}")
    print(f"  min_year        : {src_cfg['min_year']}")
    print()

    builder = DatasetBuilder(
        output_path=ds_cfg["output_path"],
        chunk_size=ds_cfg["chunk_size"],
        overlap=ds_cfg["overlap"],
        samples_per_paper=ds_cfg["samples_per_paper"],
    )

    builder.build_from_kaggle(
        json_path=src_cfg["kaggle_metadata_path"],
        n_papers=src_cfg["n_papers"],
        categories=src_cfg["categories"],
        min_year=src_cfg["min_year"],
    )


if __name__ == "__main__":
    main()
