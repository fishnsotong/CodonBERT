# CodonBERT

This is the code for the article "XXXX". The model is inspired by ProteinBERT and build by Lili Jiang@NENU.


## Table of Contents

- [CodonBERT](#codonbert)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Dependencies](#dependencies)
  - [Usage](#usage)
    - [For prediction](#for-prediction)
    - [For metrica calculation](#for-metrica-calculation)
    - [For training](#for-training)
  - [Citation](#citation)


## Installation

### Dependencies

Here is the dependencies of 
```
sklearn=1.2.1
pandas=1.5.3
numpy=1.23.5
torch=1.12.0+cu102
Biopython=1.81
tqdm=4.65.0
tensorboardX=2.6
pandas=1.4.3
emboss=6.6.0
RNAfold
```

We can install it manually by using the commands below:
```bash
conda create env -f codonbert.yaml -n Codon_Bert
conda activate Codon_Bert
git clone https://github.com/FPPGroup/CodonBERT.git
cd CodonBERT
```

## Usage
Here is the pipeline for processing, encoding data, and prediction.


### For prediction
```bash
Download the trained model
链接：https://pan.baidu.com/s/1_fTWgylKz9IjP0EIzPyBgQ 
提取码：kz65
```
```bash
python prediction.py -f <where-is-protein.fasta> -o <target-dir-to-out.fasta>
```
#### example
```bash
python prediction.py -f /mnt/public2/jiangl/Projects/CodonBERT/test_data/test_five.fasta -o /mnt/public2/jiangl/Projects/CodonBERT/test_data/result_data/test_five_result.fasta
```
### For metrica calculation
Four indicators of mRNA sequence: CAI MFE ENC GC, are calculated and stored in CSV
```bash
python get_metrics.py -e "env_path" -f 'XXX.fasta' -o "XXX.csv"
```
#### options:

```bash
  -h, --help            show this help message and exit
  -e ENV_PATH, --env_path ENV_PATH
                        environment absolute path
  -f FASTA, --fasta FASTA
                        mRNA fasta
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        metrics result path
```

#### example
```bash
python get_metrics.py -e /mnt/public2/jiangl/miniconda3/envs/RNA_index_cal -f /mnt/public2/jiangl/Projects/Project_plm_codon_optim/data/processed_output_data/fasta_file/epoch320_5_out_fix.fasta -o /mnt/public2/jiangl/Projects/Project_codon_optim/data/index_result/all_seq/epoch320_5_out_fix_result.csv
```

### For training
```bash
python train.py -i <where-is-mRNA-seq.npy> 
```
#### example
```bash
python train.py -i /mnt/public2/jiangl/Projects/Project_plm_codon_optim/data/raw_data/TPM/train_data/csd_kidney_high_TPM.npy
```


## Citation


-------



### step 1 preprocess



### step 2 encoding
We use `XX.py` to convert 
```bash

```

### step 3 prediction




# By Lily


## 一、数据预处理
数据收集：
```
带有mRNA在不同组织中表达量的数据
https://www.proteinatlas.org/download/transcript_rna_tissue.tsv.zip
```

```
和proteinatlas中mRNA（enstid） 对应的序列数据
https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_43/gencode.v43.pc_translations.fa.gz
```
数据筛选：
```
整理，截取后的mRNA序列名所对应的序列数据
all_seq.csv mRNA序列名（enstid），氨基酸序列（20-500），mRNA序列（cds）
```

```bash

python data_preprocess.py --tissue <where-is-transcript_rna_tissue.tsv.zip> --seq <where-is-gencode.v43.pc_translations.fa.gz> --out <target-is-all_seq.csv>
```


### Requirement
```
numpy
pandas
```

### usage
```
筛选高表达数据
python select_high_TPM_data.py
```

## 二、数据编码
功能：cds_seq_to_input
### Requirement
```
numpy
pandas
```

### usage
```
python proteinbert_torch_input_encode.py
py123:cds_file_path
py180~185:output_file_path
```

## 三、模型训练
### Requirement
```
protein-bert-pytorch
sklearn=1.2.1
pandas=1.5.3
numpy=1.23.5
torch=1.12.0+cu102
Biopython=1.81
tqdm=4.65.0
tensorboardX=2.6
```
### usage
```
python proteinbert_torch_pretain.py
py218~225:输入数据
py296~298:SummaryWriter设置
py316:save model path
```

## 四、模型测试
### usage
```
#模型结果还原成氨基酸序列和mRNA序列 and fix错误预测位点
python torch_model_eval.py
py165:model路径
py258:预测结果保存路径
py368:fix结果保存路径

```

## 五、mRNA序列指标计算
功能：计算优化后序列的四个指标：CAI MFE ENC GC，并存储到CSV中


### usage

```
python get_metrics.py -e "env_path" -f 'XXX.fasta' -o "XXX.csv"
```

### options:

```
  -h, --help            show this help message and exit
  -e ENV_PATH, --env_path ENV_PATH
                        environment absolute path
  -f FASTA, --fasta FASTA
                        mRNA fasta
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        metrics result path
```

### example
```
python get_metrics.py -e "/mnt/public2/jiangl/miniconda3/envs/RNA_index_cal" -f "/mnt/public2/jiangl/Projects/Project_plm_codon_optim/data/processed_output_data/fasta_file/epoch320_5_out_fix.fasta" -o "/mnt/public2/jiangl/Projects/Project_codon_optim/data/index_result/all_seq/epoch320_5_out_fix_result.csv"
```


                        


