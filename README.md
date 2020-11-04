# Exploring and Evaluating Attributes, Values, and Structures for Entity Alignment

The code of our EMNLP2020 paper *Exploring and Evaluating Attributes, Values, and Structures for Entity Alignment*.

## Dependencies

* Python 3
* [PyTorch >= 1.0](https://pytorch.org/get-started/locally/)
* [Scikit Learn](https://scikit-learn.org/stable/)
* [huggingface/transformers == 1.1.0](https://github.com/huggingface/transformers)

## Code

1. Run the following script to train all the subgraphs.

```bash
# Example
>> python train_subgraph.py --gpu_id 0 --channel Literal --dataset DBP15k/zh_en --load_hard_split [or not]
```

2. Run the following script for ensemble.

```bash
# Example
>> python ensemble_subgraphs.py --gpu_id 0 --dataset DBP15k/zh_en --svm [or not] --load_hard_split [or not]
```

Channels: {Digital, Literal, Structure, Name}

Datasets: {DBP15k/zh_en, DBP15k/fr_en, DBP15k/ja_en, DWY100k/wd_dbp, DWY100k/yg_dbp}

## Datasets

Download the datasets from [OneDrive](https://1drv.ms/u/s!AuQRz5abAH5T2jDOmiMlkqFP8s0Z?e=V6wNWS) and unzip it under the current folder.

## Hard Experimental Setting

The hard experimental setting aims to provide a more objective evaluation of the entity alignment task. We build a hard split of existing datasets and put seed entities with very different name in the test set.

Download only the hard split of DBP15k from [OneDrive](https://1drv.ms/u/s!AuQRz5abAH5T3EWhCpZrw24jTOrm?e=ufjzfW).

## Reference

If you use the code, please cite our paper:

```bib
@inproceedings{liu2020exploring,
  title={Exploring and Evaluating Attributes, Values, and Structures for Entity Alignment},
  author={Liu, Zhiyuan and Cao, Yixin and Pan, Liangming and Li, Juanzi and Liu, Zhiyuan and Chua, Tat-Seng},
  booktitle={EMNLP},
  year={2020}
}
```

## Acknowledgement
This research is supported by the National Research Foundation, Singapore under its International Research Centres in Singapore Funding Initiative. Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not reflect the views of National Research Foundation, Singapore.
