# 100-Days-Of-ML-Challenge
## Siraj Raval has challenged the machine learning/deep learning community to commit to 100 days of machine learning. Challenge accepted!
## Remark Here!
I re-uploaded my files (Day 1 & Day2 ) in Day 3. Beacause of rename some files. 
------------------------------------------------------------------------------------------------------------------------------
### Day 1 -> Data Preprocessing
> Use Imputer from sklearn method to handle the missing data.
strategy : string, optional (default=”mean”)
The imputation strategy.

If “mean”, then replace missing values using the mean along the axis.
If “median”, then replace missing values using the median along the axis.
If “most_frequent”, then replace missing using the most frequent value along the axis.
axis : integer, optional (default=0)

The axis along which to impute.

If axis=0, then impute along columns.
If axis=1, then impute along rows.

> sklearn.preprocessing.OneHotEncoder
encode categorical integer features using a one-hot aka one-of-K scheme.

> sklearn.preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True
> Simple Liner Regression Model is used for One variable, denoted x, is regarded as the predictor, explanatory, or independent variable.
The other variable, denoted y, is regarded as the response, outcome, or dependent variable.
In Data Visualization we use matplotlib , We spilt the XTrain and XTest.We only use X variable because X is independent variable.
------------------------------------------------------------------------------------------------------------------------------
### Day 2 -> Logistic Regression
> Although I added the note yesterday but some how I forgot to Click Commit Changes. But you can see my code how multipe logistic regression worked.
------------------------------------------------------------------------------------------------------------------------------
### Day 3 -> Apriori Algorithm
https://www.hackerearth.com/blog/machine-learning/beginners-tutorial-apriori-algorithm-data-mining-r-implementation/
> Aprior Algorithm is one of the Data Mining Algorithms used in recommendations such as amzon web service. I read the detail of the Alg from the link that I mentioned earlier.
> The error I found here is I used with anaconda cloud and jupyter notebook, I can't import packages to anacond because of unstastified resources.
> The next thing is I have learned in Day 3 is Clustering. In clustering , there is K - Means and Hierarchical Clustering.
The most well known is K - Means ALG. In the below, I've attached a link that explain well and thoroughly abut K - Means.
K-means is  one of  the simplest unsupervised  learning  algorithms.
https://www.datascience.com/blog/k-means-clustering
https://sites.google.com/site/dataclusteringalgorithms/k-means-clustering-algorithm

### Day 4 -> KNN (useful for classification or regression)
> KNN is used for classification and regression Problems. It is also used in data mining prblems. In Data Mining , there is Euclidean, Mahantan, and Minkowsi.
> Steps Involved in KNN
Choose the number of K of neighbours
Take the K nearest neighbours of the new data point, according to Euclidean distance
Among these K neighbours, count the number of data points in each category
Assign the new data point to the category where you counted the most neighbours
Your Model is Ready.
For Further Learning , Check out this article!!
https://medium.com/@adi.bronshtein/a-quick-introduction-to-k-nearest-neighbors-algorithm-62214cea29c7
> What is standard scaler that I used in the code?
Standardize features by removing the mean and scaling to unit variance

Centering and scaling happen independently on each feature by computing the relevant statistics on the samples in the training set. Mean and standard deviation are then stored to be used on later data using the transform method.

Standardization of a dataset is a common requirement for many machine learning estimators: they might behave badly if the individual feature do not more or less look like standard normally distributed data (e.g. Gaussian with 0 mean and unit variance).
>> A few Applications and Examples of KNN
Credit ratings — collecting financial characteristics vs. comparing people with similar financial features to a database. By the very nature of a credit rating, people who have similar financial details would be given similar credit ratings. Therefore, they would like to be able to use this existing database to predict a new customer’s credit rating, without having to perform all the calculations.
Should the bank give a loan to an individual? Would an individual default on his or her loan? Is that person closer in characteristics to people who defaulted or did not default on their loans?
In political science — classing a potential voter to a “will vote” or “will not vote”, or to “vote Democrat” or “vote Republican”.
More advance examples could include handwriting detection (like OCR), image recognition and even video recognition.

>> class sklearn.multiclass.OneVsRestClassifier(estimator, n_jobs=1)means One vs All which is used in multi - regression
>> Also known as one-vs-all, this strategy consists in fitting one classifier per class. For each classifier, the class is fitted against all the other classes. In addition to its computational efficiency (only n_classes classifiers are needed), one advantage of this approach is its interpretability. Since each class is represented by one and one classifier only, it is possible to gain knowledge about the class by inspecting its corresponding classifier. This is the most commonly used strategy for multiclass classification and is a fair default choice.

This strategy can also be used for multilabel learning, where a classifier is used to predict multiple labels for instance, by fitting on a 2-d matrix in which cell [i, j] is 1 if sample i has label j and 0 otherwise.
> from sklearn.cross_validation import train_test_split # cross validation is used for train test split
