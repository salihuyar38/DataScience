import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 


prices =pd.read_csv("GasPrices.csv")

numerics =prices.describe()

grouped_averages = prices.groupby('Name')[['Price', 'Income', 'Pumps']].mean()



#plt.boxplot([grouped_averages['Price'], grouped_averages['Income'], grouped_averages['Pumps']],
         #   labels=['Price', 'Income', 'Pumps'])
plt.boxplot(grouped_averages['Price'])
plt.title('Statistical information about Price ')
plt.show()
plt.boxplot(grouped_averages['Income'])
plt.title('Statistical information about Income ')
plt.show()
plt.boxplot(grouped_averages['Pumps'])
plt.title('Statistical information about Pumps ')
plt.show()
#print(grouped_averages['Price'].values)
#grouped_averages['Price'].values 
#gives only values from Price column and Income column 
matrix =np.array([grouped_averages['Price'].values,grouped_averages['Income'].values])



def learn_simple_regression(Matrix):
        #used my code from last week 
        x = Matrix[:,0]#first column 
        y =Matrix[:,1]#second column 
        #implement the algorithm
        N =  len(Matrix)
        sum_x =0
        sum_y =0
        for i in range(N) : 
             sum_x+= x[i] 
        x_ =sum_x / N
        for i in range(N) : 
            sum_y+= y[i] 
        y_ =  sum_y / N
        B1_num =0
        for i in range(N) : 
             B1_num+=  (x[i] - x_) * (y[i]-y_)
        B1_deno =0 
        for i in range(N): 
            B1_deno += np.square((x[i]-x_))
        B1 = B1_num / B1_deno 
        B0 =y_ - (B1 * x_)
        return (B0,B1)

f_predict =learn_simple_regression(matrix)




#imlement 2. Algorithm 

def predict_simple_regression(Matrix,match): 
        b0,b1 =match
       
        x = Matrix[:,0]
        y =b0 + (b1*x)
        return y 

'''
plt.scatter(matrix[:, 0], matrix[:, 1], color='blue', label=' Points')
x_values = np.linspace(np.min(matrix[:, 0]), np.max(matrix[:, 0]), 100)
y_values = f_predict[0] + f_predict[1] * x_values
plt.plot(x_values, y_values, color='red', label='Regression Line')
plt.xlabel('Price')
plt.ylabel('Income')
plt.title('Prediction Line for Income')
plt.legend()

income_min = np.min(matrix[:, 1])
income_max = np.max(matrix[:, 1])
normalized_income = (matrix[:, 1] - income_min) / (income_max - income_min)


normalized_matrix = np.array([matrix[:, 0], normalized_income]).T



f_predict_normalized = learn_simple_regression(normalized_matrix)
x_values = np.linspace(np.min(matrix[:, 0]), np.max(matrix[:, 0]), 100)
plt.scatter(matrix[:, 0], normalized_income, color='blue', label=' Points')
y_normalized_values = f_predict_normalized[0] + f_predict_normalized[1] * x_values
plt.plot(x_values, y_normalized_values, color='red', label='Normalized Prediction Line')
plt.xlabel('Price')
plt.ylabel('Normalized Income')
plt.title('Prediction Line for Normalized Income')
plt.legend()



plt.tight_layout()
plt.show()
'''