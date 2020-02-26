# Machine learning Classifiers for Wealth Management Dataset

This repo consists of data visualization project done for wealth management dataset from Kaggle. I have used various Machine Learning classifiers to calculate accuracy and precision to determine which model works best for this dataset. The agenda of this project is to analyze the trend of customer churn from a wealth management company.

Challenges Faced and Preliminary actions taken
•	Identify primary key columns like ID, email and other irrelevant columns like name, surname etc and delete them, as these columns don’t add value while analyzing data or classifying data.
•	Identifying null values in certain columns, and then delete those rows or add some dummy value for that column in a row depending upon the severity of the row data.
•	Identifying the datatype for each columns. If categorical data, then we have to handle it carefully and convert into numeric using various techniques like ordinal and one-hot depending on how we will compute the data.
•	Identifying the various classifiers that will fit for the use case.
•	Identifying the libraries required to implement those classifiers.
•	Installing all the required packages to support various libraries and operations. 


Dataset used:
The dataset used for this exercise can be found here https://www.kaggle.com/filippoo/deep-learning-az-ann

Data Visualization:
Before using different Classifiers we will first visualize the data and how the features in dataset directly affects the customer churn. Here I am using ‘matplotlib’ and ‘seaborn’ libraries for plotting and visualizing. 
1.	Number of Products customers own vs Total number of customers
2.	Has Credit Card vs Total number of customers
3.	Is an active member vs Total number of customers
 


4.	Age vs Balance
5.	Age vs Credit Score
 

6.	Number of Products customers own vs Age
7.	Has Credit Card vs Age
8.	Is an active member vs Age


Model Visualization:
For better understanding, we will now visualize different models based on their prediction errors. We have already compared the Accuracy score in the above table. Now we will compare the precision score as well to help us determine the best model for the Wealth Management Dataset. I have used Yellowbrick visualizer to visualize different classification models. This visualizer uses ‘sci-kit learn’ and ‘matplotlib’ libraries for visualizing.

Steps for visualizing using Yellowbrick packages:
Step 1: The first step for this will be to install yellow brick in your system:
 

Step 2: Now launch the jupyter notebook from your terminal
 

Step 3: Now import the following files from yellowbrick package
	 
    

Now we can use the code from TestCode.py for creating training and testing datasets and perform visualization.

Note:  0 – Customers who did not Exit
           1 – Customers Exited


1.	Logistic Regression:
The figure below illustrates the actual and the error in predicting number of customers exited
 

The figure below illustrates the Precision Recall Curve
Avg Precision Score: 0.74	 









2.	Gaussian Naïve Bayes
The figure below illustrates the actual and the error in predicting number of customers exited
 

The figure below illustrates the Precision Recall Curve
Avg Precision Score: 0.84
 













3.	Decision Tree
The figure below illustrates the actual and the error in predicting number of customers exited

 


The figure below illustrates the Precision Recall Curve
Avg Precision Score: 0.78
 








4.	SVM
The figure below illustrates the actual and the error in predicting number of customers exited
 


The figure below illustrates the Precision Recall Curve
Avg Precision Score: 0.85
 










5.	Random Forest
The figure below illustrates the actual and the error in predicting number of customers exited
 


The figure below illustrates the Precision Recall Curve
Avg Precision Score: 0.95
 







Factors to be considered before choosing any Machine learning algorithm:

 



Conclusion
We choose Random Forest Classifier for the Wealth Management Dataset because of the following reasons:
•	It yields the Highest Accuracy Sccore – 0.864
•	It yields the Highest Precision Value – 0.95
•	The error prediction is relatively low as compared with other classifying models.

Additional interesting facts about random forest why and how it fits best for our dataset:
•	Random forest works great in financial sector to predict loyal customers and fraud customers
•	Helps in predicting the disease by analyzing patients medical records.
•	Random forest is also used in e-commerce websites to identify the likelihood of a customer and predicting items based on the likelihood.
•	Considered as a very strong algorithm for accurate prediction based on the inputs given.




References
https://www.dataquest.io/blog/sci-kit-learn-tutorial/
https://www.youtube.com/watch?v=J4Wdy0Wc_xQ
https://www.youtube.com/watch?v=D_2LkhMJcfY
https://www.kaggle.com/nasirislamsujan/bank-customer-churn-prediction
https://stackabuse.com/implementing-svm-and-kernel-svm-with-pythons-scikit-learn/
https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html
https://www.scikit-yb.org/en/latest/api/classifier/class_prediction_error.html
