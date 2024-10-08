# Datasets

Here are brief descriptions of the datasets in this folder.

## `ames-housing-data`

[ames-housing-data](./ames-housing-data/)

Housing description with 81 features including sale price. Could be used for 
decision tree regression problems.

Referenced in [Lecture 17 -- Regression-I-Linear](../17-Regression-I-Linear.qmd).

## `images`

[images](./images/)

Contains one file called `boat.dat`. Referenced in 
[Lecture 10 -- Low Rank and SVD](../10-Low-Rank-and-SVD.qmd).

## Iris

Although not committed to the repo, the Iris dataset is loaded via 
`sklearn.datasets.load_iris()` in [15-Classification-II-kNN](../15-Classification-II-kNN.qmd)
as a comparison with kNN.

Also used in [16-Classification-III-NB-SVM.qmd](../16-Classification-III-NB-SVM.qmd).

Used for both **kNN** and **Decision Trees** examples.

## `MNIST_data`

[MNIST_data](./MNIST_data/)

Used in [25-NN-III-CNNs.qmd](../25-NN-III-CNNs.qmd).

Also referenced in [15-Classification-II-kNN.qmd](../15-Classification-II-kNN.qmd),
but is loaded via `sklearn.datasets.load_digits()`.

## `net-traffic`

[net-traffic](./net-traffic/)

Used in [10-Low-Rank-and-SVD.qmd](../10-Low-Rank-and-SVD.qmd).

## `social`

[social](./social/)

Used in [12-Anomaly-Detection-SCD-III.qmd](../12-Anomaly-Detection-SVD-III.qmd).

## `titanic`

[titanic](./titanic/)

## `ats-admissions.csv`

[ats-admissions.csv](./ats-admissions.csv)

Used in [18-Regression-II-Logistic.qmd](../18-Regression-II-Logistic.qmd) with
**logistic regression** to model the probability of getting admitted into grad school.

## `cal_housing.*`

[cal_housing.data](./cal_housing.data) and [cal_housing.domain](./cal_housing.domain).

Referenced in [](../02C-Sklearn.qmd), but seems to be downloading it with
`sklearn.datasets.fetch_california_housing`. **Linear regression** model is used.

Not clear if this the local copy is being used.

## `football.*`

[football.gml](./football.gml) -- Graph Modeling Language

Used in [21-Networks-I.qmd](../21-Networks-I.qmd).

[football.txt](./football.txt)

Both the `.gml` and `.txt` versions used in [22-Networks-II-Centrality-Clustering.qmd](../22-Networks-II-Centrality-Clustering.qmd).


## `ncaa-fb-scores-2020.csv`

[ncaa-fb-scores-2020](./ncaa-fb-scores-2020.csv)

Not currently used.

## `HorseKicks.txt`

[HorseKicks.txt](./HorseKicks.txt)

Used in [03-Probability-and-Statistics-Refresher.qmd](../03-Probability-and-Statistics-Refresher.qmd)
to demonstrate Poisson distribution and in [06-Clustering-I-kmeans.qmd](../06-Clustering-I-kmeans.qmd)
to talk about scaling of features, but might be removed.

## `polblogs.gml`

Not currently used.

## `review.json`

Not currently used.

## `vertebrate.csv`

[vertebrate.csv](./vertebrate.csv)

Downloaded from the Intro DM book website [here](https://www.cse.msu.edu/~ptan/dmbook/software/).

Table with 16 rows, each of a different animal species, and 7 traits, the last
being categorization as {mammals, reptiles, fishes, amphibians, birds}. Can be
used in **Decision Trees**, for example, to classify as mammal/non-mammal.

Likely used in Decision Trees lecture.


## `wine.data`

[wine.data](./wine.data)

Used in
[16-Classification-III-NB-SVM.qmd](../16-Classification-III-NB-SVM.qmd).

