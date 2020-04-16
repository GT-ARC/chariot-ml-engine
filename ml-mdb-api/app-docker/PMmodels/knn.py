import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.font_manager

from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from scipy.stats import scoreatpercentile

class KNN(object):

    def __init__(self):
        self.method = 'median'
        self.radius = 1.0
        self.algorithm = 'auto'
        self.metric = 'minkowski'
        self.p = 2
        self.metric_params = None
        self.n_jobs = 1

    def fit(self, data, n_neighbors=5, leaf_size=30):
        """function to fit the model
        - threshold: used to decide the binary label
        - labels_: binary labels of training data
        :return: self
        :rtype: object
        """
        # Validate inputs X and y (optional)
        data = check_array(data)

        print ("Model fitting...")

        clf = NearestNeighbors(n_neighbors=n_neighbors,
                               radius=self.radius,
                               algorithm=self.algorithm,
                               leaf_size=leaf_size,
                               metric=self.metric,
                               p=self.p,
                               metric_params=self.metric_params,
                               n_jobs=self.n_jobs)        
        
        # KDTree is used for faster nearest neighor finding
        tree_ = KDTree(data, leaf_size=leaf_size, metric=self.metric)
        # model fitting
        clf.fit(data)

        return clf, tree_

    def testset_generation(self, data):
        # select specific columns from the loaded table
        print ("Testset_generating...")

        # Generate uniform distributed novel observations for tesing uses
        len_X_test = len(data)//10
        min_x = min(data[:, 0])
        max_x = max(data[:, 0])
        min_y = min(data[:, 1])
        max_y = max(data[:, 1])

        X_test = np.zeros([len_X_test,2])
        X_test = [elem.tolist() for elem in X_test]
        X_test = np.array(X_test)
        margin_x = max_x - min_x
        margin_y = max_y - min_y
        X_test[:, 0] = np.random.uniform(low=min_x - 0.1*margin_x, high=max_x + 0.1*margin_x, size=len_X_test)
        X_test[:, 1] = np.random.uniform(low=min_y - 0.1*margin_y, high=max_y + 0.1*margin_y, size=len_X_test)

        return X_test

    def _threshold_calculation(self, clf):
        """Internal function to calculate key attributes:
        - decision_scores: used to calculate threshold        
        - threshold: used to decide the binary label
        :return: threshold
        :rtype: object
        """

        # calculate the neighbors of each indexed point and store in dist_arr
        dist_arr, _ = clf.kneighbors(n_neighbors=self.n_neighbors, return_distance=True)

        # calculate dist for later threshold and label calculation
        if self.method == 'largest':
            dist = dist_arr[:, -1]
        elif self.method == 'mean':
            dist = np.mean(dist_arr, axis=1)
        elif self.method == 'median':
            dist = np.median(dist_arr, axis=1)

        decision_scores_ = dist.ravel()
        threshold_ = scoreatpercentile(decision_scores_, 100)

        return threshold_

    def _decision_function(self, tree_, data):
        """function to calculate the decision score for testset
        :return: list of scores
        :rtype: float
        """
        data = check_array(data)

        # initialize the output score
        pred_scores = np.zeros([data.shape[0], 1])

        for i in range(data.shape[0]):
            x_i = data[i, :]
            x_i = np.asarray(x_i).reshape(1, x_i.shape[0])

            # get the distance between the current data point and its k nearest neighbors
            dist_arr, _ = tree_.query(x_i, k=self.n_neighbors)

            if self.method == 'largest':
                dist = dist_arr[:, -1]
            # calculate the mean of each row
            elif self.method == 'mean':
                dist = np.mean(dist_arr, axis=1)
            elif self.method == 'median':
                dist = np.median(dist_arr, axis=1)

            # the last one: largest distance
            pred_score_i = dist[-1]

            # record the current item
            pred_scores[i, :] = pred_score_i

        return pred_scores.ravel()


    def predict(self, clf, tree_, testset):
        print ("Model predicting...")

        params = clf.get_params()

        self.n_neighbors = int(params['n_neighbors'])
        self.leaf_size = int(params['leaf_size'])

        pred_score = self._decision_function(tree_, testset)
        threshold_ = self._threshold_calculation(clf)
        y_pred_test =  (pred_score <= threshold_).astype('int').ravel()
        
        return y_pred_test

    def visualization(self, clf, tree_, trainset, predicted_testset):
        print ("Calculating decision boundaries...")
        params = clf.get_params()
        self.n_neighbors = int(params['n_neighbors'])
        self.leaf_size = int(params['leaf_size'])

        x, y = trainset[:, 0], trainset[:, 1]

        min_x = min(trainset[:, 0])
        max_x = max(trainset[:, 0])
        min_y = min(trainset[:, 1])
        max_y = max(trainset[:, 1])

        h = 0.02
        x_min, x_max = min_x - 1, max_x + 1
        y_min, y_max = min_y - 1, max_y + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = self._decision_function(tree_, np.c_[xx.ravel(), yy.ravel()]) * -1
        Z = Z.reshape(xx.shape)

        print ("Plotting graphs...")
        plt.figure(figsize=(12,9))
        plt.title("Novelty Detection Using KNN") 
        outliers_fraction = 0
        scores_pred = self._decision_function(tree_, trainset) * -1
        threshold = scoreatpercentile(scores_pred,
                                    100 * outliers_fraction)

        legend_box = []
        legend_title = []

        plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7), cmap=plt.cm.PuBu)
        a1 = plt.contour(xx, yy, Z, levels=[threshold], linewidths=2, linestyles='solid', colors='red')
        a2 = plt.contourf(xx, yy, Z, levels=[threshold, Z.max()], colors='orange')
        b1 = plt.scatter(trainset[:, 0], trainset[:, 1], c='black', s=20)
        legend_box.append(a1.collections[0])
        legend_box.append(b1)
        legend_title.append("learned frontier")
        legend_title.append("train observations")

        if len(predicted_testset):
            X_test_right = np.array([item for item in predicted_testset if item[2] == 1 ])
            X_test_wrong = np.array([item for item in predicted_testset if item[2] == 0 ])
            b2 = plt.scatter(X_test_right[:, 0], X_test_right[:, 1], c='blueviolet', s=40,
                             edgecolors='k')
            b3 = plt.scatter(X_test_wrong[:, 0], X_test_wrong[:, 1], c='green', s=10,
                             edgecolors='k')
            legend_box.append(b2)
            legend_title.append("new observations")


        plt.legend(legend_box,
                   legend_title,
                   loc="upper left",
                   prop=matplotlib.font_manager.FontProperties(size=11))

        plt.savefig('knn.png')
        # plt.show()

        return 0