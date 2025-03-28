#######################################################################################################################################################
# 
# Name:
# SID:
# Exam Date:
# Module:
# Github link for this assignment:  
#
#######################################################################################################################################################
# Instruction 1. Read each question carefully and complete the scripts as instructed.

# Instruction 2. Only ethical and minimal use of AI is allowed. You may use AI to get advice on tool usage or language syntax, 
#                but not to generate code. Clearly indicate how and where you used AI.

# Instruction 3. Include comments explaining the logic of your code and the output as a comment below the code.

# Instruction 4. Commit to Git and upload to ELE once you finish.

#######################################################################################################################################################

# Question 1 - Loops and Lists
# You are given a list of numbers representing weekly sales in units.
weekly_sales = [120, 85, 100, 90, 110, 95, 130]
# Calculate the average sales by dividing the total sales by the number of weeks
average_sales = sum(weekly_sales) / len(weekly_sales)
# Print the average sales for the period
print("Average sales:", average_sales)  # Display the calculated average sales

# Loop through each week's sales in the list
for sale in weekly_sales:
    # Check if the current week's sale is greater than the average sales
    if sale > average_sales:
        print(sale, "is above average sales")  # Print that the sale is above average
    else:
        print(sale, "is below average sales")  # Print that the sale is below average
# Output:
#Average sales: 104.28571428571429
#120 is above average sales
#85 is below average sales
#100 is below average sales
#90 is below average sales
#110 is above average sales
#95 is below average sales
#130 is above average sales


# Write a for loop that iterates through the list and prints whether each week's sales were above or below the average sales for the period.
# Calculate and print the average sales.

#######################################################################################################################################################

# Question 2 - String Manipulation
# A customer feedback string is provided:
customer_feedback = """The product was good but could be improved. I especially appreciated the customer support and fast response times."""

# Find the first and last occurrence of the words 'good' and 'improved' in the feedback using string methods.
# Store each position in a list as a tuple (start, end) for both words and print the list.
# Define the customer feedback string
customer_feedback = """The product was good but could be improved. I especially appreciated the customer support and fast response times."""

# Find the first occurrence of the word 'good'
first_good = customer_feedback.find("good")  # Get the starting index of "good"

# Find the last occurrence of the word 'good'
last_good = customer_feedback.rfind("good")   # Get the starting index of the last occurrence of "good"

# Find the first occurrence of the word 'improved'
first_improved = customer_feedback.find("improved")  # Get the starting index of "improved"

# Find the last occurrence of the word 'improved'
last_improved = customer_feedback.rfind("improved")   # Get the starting index of the last occurrence of "improved"

# Create tuples with (start, end) positions for each occurrence
good_first_tuple = (first_good, first_good + len("good"))          # Tuple for first occurrence of "good"
good_last_tuple = (last_good, last_good + len("good"))               # Tuple for last occurrence of "good"
improved_first_tuple = (first_improved, first_improved + len("improved"))  # Tuple for first occurrence of "improved"
improved_last_tuple = (last_improved, last_improved + len("improved"))       # Tuple for last occurrence of "improved"

# Store the tuples in a list
positions = [good_first_tuple, good_last_tuple, improved_first_tuple, improved_last_tuple]  # List containing all tuples

# Print the list of tuples
print(positions)

# Output:
#[(16, 20), (16, 20), (34, 42), (34, 42)]

#######################################################################################################################################################

# Question 3 - Functions for Business Metrics
# Define functions to calculate the following metrics, and call each function with sample values (use your student ID digits for customization).

# 1. Net Profit Margin: Calculate as (Net Profit / Revenue) * 100.
# 2. Customer Acquisition Cost (CAC): Calculate as (Total Marketing Cost / New Customers Acquired).
# 3. Net Promoter Score (NPS): Calculate as (Promoters - Detractors) / Total Respondents * 100.
# 4. Return on Investment (ROI): Calculate as (Net Gain from Investment / Investment Cost) * 100.
# For Net Profit Margin, using net_profit = 7400 and revenue = 98558

# Function to calculate Net Promoter Score (NPS): ((Promoters - Detractors) / Total Respondents) * 100
# --- function definitions MUST come first ---
def net_profit_margin(net_profit, revenue):
    return (net_profit / revenue) * 100

def customer_acquisition_cost(total_cost, new_customers):
    return total_cost / new_customers

def net_promoter_score(promoters, detractors, total):
    return ((promoters - detractors) / total) * 100

def return_on_investment(net_gain, cost):
    return (net_gain / cost) * 100

# --- then the calls ---
if __name__ == "__main__":
    print("Net Profit Margin:", net_profit_margin(7400, 98558))
    print("CAC:", customer_acquisition_cost(740, 98))
    print("NPS:", net_promoter_score(74, 9, 85))
    print("ROI:", return_on_investment(7400, 9858))

#######################################################################################################################################################

# Question 4 - Data Analysis with Pandas
# Using a dictionary sales_data, create a DataFrame from this dictionary, and display the DataFrame.
# Write code to calculate and print the cumulative monthly sales up to each month.

import pandas as pd

sales_data = {'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May'], 'Sales': [200, 220, 210, 240, 250]}

df = pd.DataFrame(sales_data) 
print ("Monthly Sales Data")
print (df)

df['Cumulative Sales'] = df ['Sales'].cumsum()
print ("Cumulative Monthly Sales:")
print(df[['Month', 'Cumulative Sales']])

#######################################################################################################################################################

# Question 5 - Linear Regression for Forecasting
# Using the dataset below, create a linear regression model to predict the demand for given prices.
# Predict the demand if the company sets the price at £26. Show a scatter plot of the data points and plot the regression line.

# Price (£): 15, 18, 20, 22, 25, 27, 30
# Demand (Units): 200, 180, 170, 160, 150, 140, 130

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Build the DataFrame
df = pd.DataFrame({
    'Price': [15, 18, 20, 22, 25, 27, 30],
    'Demand': [200, 180, 170, 160, 150, 140, 130]
})

# Fit a linear regression model
X = df[['Price']]
y = df['Demand']
model = LinearRegression().fit(X, y)

# Predict demand at £26
predicted_demand = model.predict([[26]])[0]
print(f"Predicted demand at £26: {predicted_demand:.2f} units")

# Scatter plot + regression line
plt.figure()
plt.scatter(df['Price'], df['Demand'])
x_line = np.linspace(df['Price'].min(), df['Price'].max(), 100).reshape(-1,1)
plt.plot(x_line, model.predict(x_line))
plt.xlabel('Price (£)')
plt.ylabel('Demand (Units)')
plt.title('Price vs Demand with Regression Line')
plt.show()


#######################################################################################################################################################

# Question 6 - Error Handling
# You are given a dictionary of prices for different products.
prices = {'A': 50, 'B': 75, 'C': 'unknown', 'D': 30}

# Write a function to calculate the total price of all items, handling any non-numeric values by skipping them.
# Include error handling in your function and explain where and why it’s needed.
# Define the dictionary with prices
prices = {'A': 50, 'B': 75, 'C': 'unknown', 'D': 30}

def calculate_total_price(price_dict):
    """
    Calculate the total price of all items in the dictionary.
    
    For each product, we try to convert the price to a float. If the value is not numeric,
    a ValueError or TypeError will be raised. The try/except block catches these exceptions,
    prints a message indicating that the non-numeric value is being skipped, and continues 
    processing the remaining items.
    """
    total = 0  # Initialize total sum
    for product, price in price_dict.items():
        try:
            # Attempt to convert the price to a float
            total += float(price)
        except (ValueError, TypeError):
            # This block catches errors when price is non-numeric (e.g., 'unknown')
            print(f"Skipping product '{product}' due to non-numeric price: {price}")
    return total  # Return the cumulative total of valid prices

# Call the function and display the result
total_price = calculate_total_price(prices)
print("Total Price:", total_price)

#######################################################################################################################################################

# Question 7 - Plotting and Visualization
# Generate 50 random numbers between 1 and 500, then:
# Plot a histogram to visualize the distribution of these numbers.
# Add appropriate labels for the x-axis and y-axis, and include a title for the histogram.

import matplotlib.pyplot as plt
import numpy as np
# Generate 50 random integers between 1 and 500
random_numbers = np.random.randint(1, 501, size=50)

# Create a histogram to visualize the distribution of the random numbers
plt.hist(random_numbers, bins=10, edgecolor='black')

# Add labels and title to the plot
plt.xlabel('Random Numbers')
plt.ylabel('Frequency')
plt.title('Histogram of 50 Random Numbers between 1 and 500')

# Display the histogram
plt.show()

#######################################################################################################################################################

# Question 8 - List Comprehensions
# Given a list of integers representing order quantities.
quantities = [5, 12, 9, 15, 7, 10]
# Use a list comprehension to double each quantity that is 10 or more.
# For each quantity 'q' in the list, if q is greater than or equal to 10, multiply it by 2, otherwise keep it the same.
doubled_quantities = [q * 2 if q >= 10 else q for q in quantities]
# Print the original list of quantities
print("Original quantities:", quantities)
# Print the new list of quantities with doubled values where applicable
print("Modified quantities:", doubled_quantities)


# Use a list comprehension to create a new list that doubles each quantity that is 10 or more.
# Print the original and the new lists.

#######################################################################################################################################################

# Question 9 - Dictionary Manipulation
# Using the dictionary below, filter out the products with a rating of less than 4 and create a new dictionary with the remaining products.
ratings = {'product_A': 4, 'product_B': 5, 'product_C': 3, 'product_D': 2, 'product_E': 5}
# Use a dictionary comprehension to filter out products with a rating less than 4.
filtered_ratings = {product: rating for product, rating in ratings.items() if rating >= 4}

# Print the new dictionary with the remaining products.
print("Filtered Products (rating >= 4):", filtered_ratings)
#######################################################################################################################################################

# Question 10 - Debugging and Correcting Code
# The following code intends to calculate the average of a list of numbers, but it contains errors:
values = [10, 20, 30, 40, 50]
total = 0
for i in values:
    total = total + i
average = total / len(values)
print("The average is" + average)

# Identify and correct the errors in the code.
# Comment on each error and explain your fixes.

# Define the list of values
values = [10, 20, 30, 40, 50]

# Initialize total to 0 to accumulate the sum of values
total = 0

# Loop through each number in the values list and add it to total
for i in values:
    total = total + i  # This correctly accumulates the total sum

# Calculate the average by dividing the total sum by the number of values
average = total / len(values)

# Error: Cannot concatenate a string and a float directly.
# Fix: Convert 'average' to a string or separate items using a comma in print.
print("The average is " + str(average))
#The average is 30.0


#######################################################################################################################################################
