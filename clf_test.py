'''
this example is an altered version of:
    https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
'''

import sys
from math import sqrt
import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification


def close_event():
    plt.close()


def test(classifier_name, trial_number):

    trial_nb = int(trial_number)
    
    ''' Read the parameters for the chosen classifier and instantiate the model '''
    if classifier_name == "Nearest Neighbors":
        params = yaml.safe_load(open('params.yaml'))['classifiers'][classifier_name]
        n = params['n_neighbors']
        algo = params['algorithm']
        clf = KNeighborsClassifier(n_neighbors=n, algorithm=algo)
        
    elif classifier_name == "SVC":    
        params = yaml.safe_load(open('params.yaml'))['classifiers'][classifier_name]
        gamma = params['gamma']
        C = params['C']
        kern = params['kernel']
        clf = SVC(kernel=kern, gamma=gamma, C=C)
        
    elif classifier_name == "MLP":    
        params = yaml.safe_load(open('params.yaml'))['classifiers'][classifier_name]
        alpha = params['alpha']
        max_iter = params['max_iter']
        clf = MLPClassifier(alpha=alpha, max_iter=max_iter)
    
    else:
        print("this classifier name not recognized... exiting.")
        sys.exit()
    
    
    ''' Generate a random n-class classification problem '''
    # read the dataset generation parameters
    params = yaml.safe_load(open('params.yaml'))['dataset']
    ns = params['n_samples']
    nf = params['n_features']
    nr = params['n_redundant']
    ni = params['n_informative']
    rs = params['random_state']
    nc = params['n_clusters_per_class']
    cs = params['class_sep'] * sqrt(2/trial_nb) # this should give about 20 trials before
                                                # the classes become indistiguishable
    print("trial number:", trial_number, " --- ", "class separtion:", f'{cs:0.3f}', end=' --- ')
    
    X, y = make_classification(n_samples=ns, n_features=nf, n_redundant=nr, n_informative=ni,
                               random_state=rs, n_clusters_per_class=nc, class_sep=cs)
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)
    
    h = .02  # step size in the mesh (only for plotting)
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    
    ''' the classifier fit '''
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print("score:", score)
    
    
    ''' Plot the results '''
    # plot the dataset first
    figure=plt.figure()
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot()
    ax.set_title("classifier: " + classifier_name + ",  score: " + str(score))
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                edgecolors='k')
    # Plot the testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
               edgecolors='k')
    
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
    
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k')
    # Plot the testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
               edgecolors='k', alpha=0.6)
    
    timer = figure.canvas.new_timer(interval = 2000) # close plot window after N milliseconds
    timer.add_callback(close_event)
    timer.start()
    plt.show()
    #plt.savefig(str(trial_number) + ".png")

    return score

