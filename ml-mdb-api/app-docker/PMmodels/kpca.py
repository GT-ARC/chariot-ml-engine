from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
import numpy as np
from numpy import linalg as la
import random
import matplotlib.pyplot as plt
import matplotlib.font_manager


class KPCA(object):

    def fit(self, data, alpha=0.3, gamma=4):
        scaler = StandardScaler().fit(data)
        data = scaler.transform(data)

        # if len(data)>5000:
        #     index = np.random.choice(len(data), len(data)//3)
        #     data = data[index]        
        
        print ("Model fitting...")

        clf = KernelPCA(n_components = 100, kernel="rbf", fit_inverse_transform=True, alpha=alpha, gamma=gamma)
        clf.fit(data)
        max_err = self._err_calc(clf, data).max()

        return clf, scaler, max_err

    def _err_calc(self, clf, data):
        X_kpca = clf.transform(data)
        X_back = clf.inverse_transform(X_kpca)
        loss=la.norm(data - X_back,axis=1)

        return loss

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


    def predict(self, clf, scaler, max_err, data):
        print ("Model predicting...")

        data = scaler.transform(data)
        error = self._err_calc(clf, data)

        length = len(data)
        y_pred_test = []

        for i in range(length):
            if error[i] > max_err:
                y_pred_test.append(0)
            else:
                y_pred_test.append(1)
        
        return np.array(y_pred_test)

    def visualization(self, clf, scaler, max_err, trainset, predicted_testset):
        print ("Calculating decision boundaries...")

        trainset = scaler.transform(trainset)
        predicted_testset[:, :2] = scaler.transform(predicted_testset[:, :2])

        x, y = trainset[:, 0], trainset[:, 1]

        min_x = min(trainset[:, 0])
        max_x = max(trainset[:, 0])
        min_y = min(trainset[:, 1])
        max_y = max(trainset[:, 1])

        range_x = max_x - min_x
        range_y = max_y - min_y

        m = 100
        step_x = (1.0)*range_x *1.3/(m-1)
        step_y = (1.0)*range_y *1.3/(m/2-1)

        x_min, x_max = min_x - range_x*0.15, max_x + range_x*0.15 + 1e-6
        y_min, y_max = min_y - range_y*0.15, max_y + range_y*0.15 + 1e-6

        xx, yy = np.meshgrid(np.arange(x_min, x_max, step_x),
                             np.arange(y_min, y_max, step_y))

        err = np.zeros([m//2,m])

        for i in range(m//2):
            for j in range(m):
                x_in = np.array([xx[0,j],yy[i,0]])
                err[i,j] = self._err_calc(clf, x_in.reshape(1, -1))

        Z = err

        print ("Plotting graphs...\n")        
        plt.figure(figsize=(12,9))
        plt.title("Novelty Detection Using Kernel PCA")        

        legend_box = []
        legend_title = []

        plt.contourf(xx, yy, Z, levels=np.linspace(max_err, Z.max(), 7), cmap=plt.cm.PuBu)
        a1 = plt.contour(xx, yy, Z, levels=[max_err], linewidths=2, linestyles='solid', colors='red')
        a2 = plt.contourf(xx, yy, Z, levels=[Z.min(), max_err], colors='orange')
        b1 = plt.scatter(x, y, c='black', s=20)
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

        plt.savefig('kpca.png')
        # plt.show()

        return 0