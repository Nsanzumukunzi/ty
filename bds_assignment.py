from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np

def cross_validation_polyfit(x_data, y_data, degree):
    x = x_data.flatten()
    y = y_data.flatten()
    
    results = {}
    
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

    coeffs = np.polyfit(x_train, y_train, degree)

    model = np.poly1d(coeffs)
    
    ####### Testing Model #######
    # fit values, and mean
    yhat = model(x_test)                         #predicted
    ybar = np.sum(y_test)/len(y_test)          
    ssreg = np.sum((yhat-ybar)**2)   
    sstot = np.sum((y_test - ybar)**2)    
    
    rmse =np.sqrt(mean_squared_error(y_test, yhat))
    
    adj_rsquared = 1 - (1 - (ssreg / sstot)) * (len(y_test) - 1) / (len(y_test) - x_test[1] - 1)
    rsquared = r2_score(y_test, yhat)
    
    results['RSquared'] = rsquared
    results['RMSE'] = rmse
    results['Adj_RSquared'] = adj_rsquared

    return results


def polyfit(x, y, degree):
    results = {}

    coeffs = np.polyfit(x, y, degree)

    # r-squared
    p = np.poly1d(coeffs)
    # fit values, and mean
    yhat = p(x)                         # or [p(z) for z in x]
    ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    
    rmse = np.sqrt(mean_squared_error(y, yhat))
    adj_rsquared = 1 - (1 - (ssreg / sstot)) * (len(y) - 1) / (len(y) - x[1] - 1)
    rsquared = r2_score(y, yhat)
    
    results['RSquared'] = rsquared
    results['RMSE'] = rmse
    results['Adj_RSquared'] = adj_rsquared

    return results
