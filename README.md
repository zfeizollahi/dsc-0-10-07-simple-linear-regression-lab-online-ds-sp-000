
# Simple Linear Regression - Lab

## Introduction

Regression analysis forms the basis of machine learning experiments. Understanding regression will help you to get the foundations of most machine learing algorithms. Ever wondered what's at the heart of an artificial neural network processing unstructured data like music and graphics? It can be linear regression!


## Objectives

You will be able to:

* Calculate the slope of a line using standard slope formula 
* Calculate the y-intercept using the slope value
* Draw a regression line based on calculated slope and intercept
* Predict the label of a previously unseen data element



## Let's get started 

A first step towards understanding regression is getting a clear idea about "linear" regression and basic linear algebra. 

The calculation for the best-fit line's slope, `m` is calculated as :
![](best-fit-slope.png)


As in our previous lesson, let's break down the formula into its parts. First we shall import the required libraries and define some data points to work with. We shall first create some toy data as numpy arrays. Let's do this for you to give you a head start. 


```python
# import necessary libraries

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

# Initialize vectors X and Y with given values and create a scatter plot
X = np.array([1,2,3,4,5,6,8,8,9,10], dtype=np.float64)
Y = np.array([7,7,8,9,9,10,10,11,11,12], dtype=np.float64)
```

## Show a scatter plot between X and Y and comment on the output 


```python
# Scatter plot
plt.scatter(X, Y)
```




    <matplotlib.collections.PathCollection at 0x7fb2e82af278>




![png](index_files/index_6_1.png)


#### Your observations about relationship in X and Y 

The data looks fairly linearly correlated.

In a data analysis context, we can think of these points as two vectors:

* **vector X**: the features of our model
* **vector Y**: the labels for given features

## Write a function calc_slope()

Write a function `calc_clope()` that takes in x and y vectors and calculates the slope using the formula shown above. 


```python
# Write the function to calculate slope as: 
# (mean(x) * mean(y) – mean(x*y)) / ( mean (x)^2 – mean( x^2))
def calc_slope(xs,ys):
    xy_s = xs * ys
    x_sq = xs ** 2
    nominator = (xs.mean() * ys.mean()) - xy_s.mean()
    denominator = (xs.mean() ** 2) - x_sq.mean()
    slope = nominator / denominator
    return slope

calc_slope(X,Y)

# 0.5393518518518512
```




    0.5393518518518512



Great, so we have our slope. Next we calculate the intercept. 

As a reminder, the calculation for the best-fit line's y-intercept is:

![](best-fit-y-intercept.png)

## Write a function best_fit()

Write a function `best_fit()` that takes in X and Y, calculates the slope using above above  and intercept using the formula. The function should return slope and intercept values. 


```python
# use the slope function with intercept formula to return calculate slop and intercept from data points

def best_fit(xs,ys):
    m = calc_slope(xs, ys)
    b = ys.mean() - m * xs.mean()
    return m, b

# Uncomment below to test your function

m, b = best_fit(X,Y)
m, b

# (0.5393518518518512, 6.379629629629633)
```




    (0.5393518518518512, 6.379629629629633)



We now have a working model with `m` and `b` as model parameters. We can create a line for the data points using the calculated slope and intercept:

* Recall that `y=mx+b`. We can now use slope and intercept values along with X data points (features) to calculate the Y data points (labels) of the regression line. 

## Write a function reg_line() 

Write a function `reg_line()` that takes in slope, intercept and X vector and calculates the regression line using Y= mX+b for each point in X.


```python
def reg_line (m, b, xs):
    return m * xs + b

# Uncomment below
regression_line = reg_line(m,b,X)
regression_line
```




    array([ 6.91898148,  7.45833333,  7.99768519,  8.53703704,  9.07638889,
            9.61574074, 10.69444444, 10.69444444, 11.2337963 , 11.77314815])



## Plot the (x,y) data points and draw the calculated regression line for visual inspection


```python
plt.plot(X, regression_line)
plt.scatter(X, Y, c='blue')
```




    <matplotlib.collections.PathCollection at 0x7fb318767358>




![png](index_files/index_16_1.png)



```python
# Plot data and regression line
```


![png](index_files/index_17_0.png)


So there we have it, our least squares regression line. This is the best fit line and does describe the data pretty well (still not perfect though). 

## Describe your Model Mathematically and in words

### Your answer here
The best fit line is the line which minimizes the least squares variance - meaning that when you add up the distance from the dots to this line, it is the line has the shortest "distance" for the sum of all the distances. 



## Predicting label for new data

So, how might you go about actually making a prediction based on this model you just made?

Now that we have a working model with m and b as model parameters. We can fill in a value of x with these parameters to identify a corresponding value of y according to our model. Recall the formula 
![](pred.jpg)

Let's try to find a y prediction for a new value of x = 7 and unknown y, and plot the new prediction with existing data 


```python
x_new = 7
y_predicted = reg_line(m,b,x_new)
y_predicted

# 10.155092592592592
```




    10.155092592592592



## Plot the prediction with actual data and regression line 


```python
plt.plot(X, regression_line)
plt.scatter(X, Y, c='blue')
plt.scatter(x_new, y_predicted)
```




    <matplotlib.collections.PathCollection at 0x7fb2e84ef978>




![png](index_files/index_23_1.png)



```python
# Plot as above and show the predicted value

```


![png](index_files/index_24_0.png)


We now know how to create our own models, which is great, but we're stilling missing something integral: how accurate is our model? This is the topic for discussion in the next lab.

## Summary

In this lesson, we learnt how we can draw a best fit line for given data labels and features, by first calculating the slope and intercept. The calculated regression line was then used to predict the label (y-value) of a previously unseen feature (x-value). The lesson uses a simple set of data points for demonstration. Students should be able to plug in other datasets and practice with predictions for accuracy. 
