# A Simple Framework for Testing Model Drift with Various ML Models  

![alt text][image0]

[//]: # (Image References)
[image0]: ./images/trials.gif

<br/>

This example uses Scikit-Learn's [make\_classification](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html) method to create 2-class datasets that begin to become indistinguishable over many trials. In the image above, note that the score, which is the mean accuracy, decreases with each trial.  

In [params.yaml](./params.yaml), both the dataset and the classifier parameters are set. The dataset value "class_sep" determines the initial class separation. This value decreases with each trial.  

The model implementations and the plotting functionality are in [clf\_test.py](./clf_test.py).  
<br/>

#### To run on your system  

Clone this repository. 

```
~$ cd model_drift_testing
```

Activate your Python Machine Learning Environment  

```
~$ python main.py
```


The program will end when the score falls below the threshold set in [main.py](./main.py)

