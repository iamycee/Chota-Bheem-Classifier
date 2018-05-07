# Chota-Bheem-Classifier
### Using Support Vector Machines to classify 4 different characters from the 'Chota Bheem' cartoon show

I attempted to classify the characters 'Bheem', 'Chutki', 'Raju' and 'Jaggu' from the popular Indian cartoon TV show called 'Chota Bheem'.

![Chutki, Bheem, Raju, Jaggu](http://www.animationxpress.com/images/Bheem-Team-Chhota-Bheem.gif)

The prediction accuracies that I am getting here aren't great.
I tried using KNN, Bagging, GridSearch-ed different paramaters, but I think due to the high dimensions + low data points (20 images per character)
My test set accuracies peak at 75% and go as low as 40% on different folds of the KFold cross validation. 

## What you should do if you want to build up on this project: 

1. Try using different classifiers
2. Use ensembling 
3. Try variations of the PCA parameters to see if it works with another set of Principal Components
4. I might be wrong in my Data Preprocessing step as well, maybe try not grayscaling the images and try different dimensions
