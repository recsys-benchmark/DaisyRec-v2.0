![DaisyRec](pics/logo.png)

![PyPI - Python Version](https://img.shields.io/badge/pyhton-3.5%2B-blue) [![Version](https://img.shields.io/badge/version-v2.0-orange)](https://github.com/recsys-benchmark/DaisyRec-v2.0) ![GitHub repo size](https://img.shields.io/github/repo-size/recsys-benchmark/DaisyRec-v2.0) ![GitHub](https://img.shields.io/github/license/recsys-benchmark/DaisyRec-v2.0)

## Overview

<!-- ![daisyRec's structure](pics/framework.png) -->

DaisyRec is a Python toolkit dealing with rating prediction and item ranking issue.

The name DAISY (roughly :) ) stands for multi-**D**imension f**A**irly compAr**I**son for recommender **SY**stem.

<p align="center">
<img src="pics/framework.png" align="center" width="100%" style="margin: 0 auto">
</p>

To get all dependencies, run:

    pip install -r requirements.txt

Before running, you need first run: 

    python setup.py build_ext --inplace

to generate `.so` or `.pyd` file used for further import.

Make sure you have a **CUDA** enviroment to accelarate since these deep-learning models could be based on it. We will consistently update this repo.

DaisyRec handled ranking issue mainly and split recommendation problem into point-wise ones and pair-wise ones so that different loss function are constructed such as BPR, Top-1, Hinge and Cross Entropy. <!--All algorithms already implemented are exhibited below:-->

<!--<img src="pics/algos.png" width="40%" height="30%" style="margin: auto; cursor:default" />-->

Use `main.py` to achieve KPI results calculated by certain algorithm above. For example, you can implement this program to implement BPR-MF:

    python main.py --problem_type=pair --algo_name=mf --loss_type=BPR --num_ng=2



## Datasets

You can download experiment data, and put them into the `data` folder.
All data are available in links below: 

  - [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/)
  - [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)
  - [MovieLens 10M](https://grouplens.org/datasets/movielens/10m/)
  - [MovieLens 20M](https://grouplens.org/datasets/movielens/20m/)
  - [Netflix Prize Data](https://archive.org/download/nf_prize_dataset.tar)
  - [Last.fm](https://grouplens.org/datasets/hetrec-2011/)
  - [Book Crossing](https://grouplens.org/datasets/book-crossing/)
  - [Epinions](http://www.cse.msu.edu/~tangjili/trust.html)
  - [CiteULike](https://github.com/js05212/citeulike-a)
  - [Amazon-Book](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Books.csv)
  - [Amazon-Electronic](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Electronics.csv)
  - [Amazon-Cloth](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Clothing_Shoes_and_Jewelry.csv)
  - [Amazon-Music](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Digital_Music.csv)
  - [Yelp Challenge](https://kaggle.com/yelp-dataset/yelp-dataset)

## TODO list

- [ ] optimize code for effiency
- [ ] weight initialization interface
- [ ] optimizer initialization interface
- [ ] a more friendly tuner gui
- [ ] make sure Item2vec work as expected

## Cite

Here is a Bibtex entry if you ever need to cite **DaisyRec** in a research paper (please keep us posted, we would love to know if Daisy was helpful to you)

```
@inproceedings{sun2020are,
  title={Are We Evaluating Rigorously? Benchmarking Recommendation for Reproducible Evaluation and Fair Comparison},
  author={Sun, Zhu and Yu, Di and Fang, Hui and Yang, Jie and Qu, Xinghua and Zhang, Jie and Geng, Cong},
  booktitle={Proceedings of the 14th ACM Conference on Recommender Systems},
  year={2020}
}

```

## Acknowledgements

We refer to the following repositories to improve our code:

 - SliM and KNN-CF parts with [RecSys2019_DeepLearning_Evaluation](https://github.com/MaurizioFD/RecSys2019_DeepLearning_Evaluation)
 - SVD++ part with [Surprise](https://github.com/NicolasHug/Surprise)
 - NGCF part with [NGCF-PyTorch](https://github.com/huangtinglin/NGCF-PyTorch)
