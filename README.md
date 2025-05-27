# FlakyAssessor

This is the replication package associated with the paper: 'FlakyAssessor: A Semantic-aware Oracle for Assessing the Correctness of Flaky Test Patches'

## Requirements

- torch>=1.12.0
- transformers>=4.20.0
- pandas>=1.4.0
- numpy>=1.21.0
- scikit-learn>=1.1.0

## Project Structure

```
FlakyAssessor/
├── README.md                       # Project description document
├── requirements.txt                # List of Python dependencies
├── per_project_train_test.py       # project-based evaluation script
├── train_test_5fold.py             # 5-fold cross-validation script
├── aggregate_kfold_results.py      # Script to aggregate 5-fold validation results
├── config/                         # Configuration files directory
├── models/                         # Model definitions
├── data_modules/                   # Data processing modules
├── helper/                         # Helper utilities
├── data/cleaned_mutation_data.csv  # Final dataset
└── nondex_script/                  # NonDex related scripts
```


## Input Files
This is a list of input files that are required to accomplish the experiments:
- data/your_data.csv: This file should contain the flaky test code, the generated patches, ground truth labels, project names. The actual path and filename are configured in config/default_config.json.
- config/default_config.json: Configuration for data paths, model parameters, and training settings.
- config/per_project.json: Specific configuration overrides for project-based validation (inherits from default).
- config/cross_validation.json: Specific configuration overrides for cross-validation (inherits from default).
- Pre-trained Model (UnixCoder):This will be downloaded automatically by the Hugging Face transformers library if a model identifier is provided and it's not available locally. Alternatively, you can provide a local path to the model.

## Replicating FlakyAssessor Experiments
Ensure you have prepared your dataset and updated the config.json (or created custom configuration files) to point to your data and specify parameters. To run the FlakyAssessor experiments, navigate to the / folder and run the following command:

### 1.Project-based Validation

```bash
python per_project_train_test.py --config_file config/per_project.json
```


### 2.5-Fold Cross-Validation

```bash
python train_test_5fold.py --config_file config/cross_validation.json
```

### 3.Nondex Validation

```bash
python ./nondex_script/main_script.py
```


