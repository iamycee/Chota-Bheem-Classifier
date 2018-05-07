from datasetOps import preprocess_images, create_dataset        
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from  sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
import numpy as np

#convert the images to grayscale, 100*100 size + store them in preprocessed_images directory
preprocess_images()

dataset = create_dataset()
X = dataset[:, :-1]
y = dataset[:, -1]
X = np.asarray(X,  dtype='float64')     #fix the datatype of X or it gives a type mismatch error


#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

kf = KFold(n_splits=10, shuffle=True)       # max k can be 80 i.e. for leave one out cross validation

for train_index, test_index in kf.split(X):

    X_train = np.array([X[i] for i in train_index])
    X_test = np.array([X[i] for i in test_index])
    y_train = np.array([y[i] for i in train_index])
    y_test = np.array([y[i] for i in test_index])
    
    pca = PCA(n_components=0.8, whiten=True).fit(X_train)        #whitening leads to less correlated PCs, 0.8 -> capture 80% variance
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    clf = SVC(class_weight='balanced')
    param_grid = {
                    'kernel': ['poly', 'linear', 'rbf'],
                    'C': [1, 10, 100, 1e3, 5e3, 1e4, 5e4, 1e5],      #harder margin as data are probably linearly separable in higher dimensions
                    'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 1],
                 }

    clf = GridSearchCV(clf, param_grid)
    clf = clf.fit(X_train_pca, y_train)
    
   

##    Un-comment these lines to print the classification report [precision, recall, f1] and the confusion matrix[TP-FP-TN-FN]

##    y_pred = clf.predict(X_test_pca)
##    names=['Bheem', 'Chutki', 'Raju', 'Jaggu']
##    print("\n\t\t---------------CLASSIFICATION REPORT---------------\n", classification_report(y_test, y_pred, target_names=names))
##
##    print("\n\t\t---------------CONFUSION MATRIX---------------\n", confusion_matrix(y_test, y_pred))

    
    print("Prediction accuracy: ", clf.score(X_test_pca, y_test))   #prints k times for k folds 



