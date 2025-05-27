# FlakyAssessor

This is the replication package associated with the paper: 'Beyond Reruns: A Heuristic Oracle for Assessing the Correctness of Flaky Test Patches'

## Requirements

- torch>=1.12.0

* transformers>=4.20.0
* pandas>=1.4.0
* numpy>=1.21.0
* scikit-learn>=1.1.0

## 项目结构

```
FlakyAssessor/
├── README.md                      # 项目说明文档
├── requirements.txt               # Python依赖包列表
├── per_project_train_test.py      # 项目级别训练和测试脚本
├── train_test_5fold.py            # 5折交叉验证训练脚本
├── aggregate_kfold_results.py     # 聚合5折验证结果脚本
├── config/                        # 配置文件目录
│   ├── default_config.json        # 默认配置文件
│   ├── per_project.json          # 项目级别配置
│   └── cross_validation.json     # 交叉验证配置
├── models/                        # 模型定义
│   ├── patch_validator.py         # 补丁验证器模型
│   └── unixcoder.py              # UnixCoder编码器实现
├── data_modules/                  # 数据处理模块
│   ├── dataset.py                # 数据集定义
│   ├── data_loader.py            # 数据加载器
│   └── collator.py               # 数据整理器
├── helper/                        # 辅助工具
│   ├── logger.py                 # 日志工具
│   ├── configure.py              # 配置管理
│   └── utils.py                  # 通用工具函数
├── data/                          # 数据目录
└── nondex_script/                # NonDex相关脚本
```

## 使用方法

### 1. 数据准备

确保您的数据集包含以下列：

- `flaky_code`: flaky 测试代码
- `generated_patch`: 生成的补丁代码
- `isCorrect`: 标签（1=正确补丁，0=错误补丁）
- `project_name`: 项目名称
- `rerun_consistency`: 基线重跑一致性结果

### 2. 配置设置

修改配置文件 `config/default_config.json` 或创建自定义配置：

```json
{
  "data": {
    "data_dir": "data/your_dataset",
    "combined_csv_file": "your_data.csv",
    "label_col": "isCorrect",
    "flaky_col": "flaky_code",
    "patch_col": "generated_patch"
  },
  "model": {
    "encoder_name": "path/to/unixcoder-base",
    "max_length": 512
  },
  "train": {
    "epochs": 100,
    "batch_size": 16,
    "learning_rate": 1e-5
  }
}
```

### 3. 基于项目验证

```bash
python per_project_train_test.py --config_file config/per_project.json
```

### 4. 5 折交叉验证

```bash
python train_test_5fold.py --config_file config/cross_validation.json
```

### 5. 聚合交叉验证结果

```bash
python aggregate_kfold_results.py --config_file config/cross_validation.json --output_dir outputs/kfold_results --num_folds 5
```

## 输出结果

训练完成后，会在输出目录生成以下文件：

- `*_overall_details_*.csv`: 详细的预测结果
- `*_per_project_summary_*.csv`: 按项目汇总的性能指标
- `*_roc_*.png`: ROC 曲线图
- `*_mean_roc_curve.png`: 平均 ROC 曲线（交叉验证）
- 模型检查点文件
