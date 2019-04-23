# -Instacart-Market-Basket-Analysis-


Overview of the Problem
In the present world, more and more people are shifting from buying groceries offline (i.e. by visiting a neighbourhood store) to online (i.e. ordering through an app). GroceryKart is one such platform which lets users order groceries online. After selecting products through the GroceryKart app, personal shoppers review the order and do the in-store shopping and delivery for the customers. 
A very classic problem faced by such e-commerce websites is understanding the purchase behaviour of a customer. If they can correctly predict which products the customers going to buy before the customer orders them, it can give them a huge advantage in terms of warehouse stocking, delivery times, marketing strategy, improving customer experience on their app etc.
Keeping that in mind, the main goal of this exercise is to use the data of GroceryKart customer orders over time, to predict which previously purchased products will be in a user’s next order. 
You will also be handling multiple datasets at once and answering important questions about the data that will test your data science skills along the way.
Data Details
Before we get to the main problem statement, it is important to have an overview of the datasets that are provided to you. Each entity (customer, product, order, aisle, etc.) has an associated unique id. Most of the files and variable names are self-explanatory.
To be read as:
Dataset Name
Fields
Description
aisles.csv
aisle_id, aisle
The aisle from which the product was ordered, examples: energy granola bars, prepared soups salads etc.
departments.csv
department_id, department
The department of the store from which a product was ordered, examples: frozen, bakery etc.
order_products_prior.csv
order_id,product_id,add_to_cart_order,reordered 
This file contains previous order contents for all customers. 'reordered' indicates that the customer has a previous order that contains the product. Note that some orders will have no reordered items. You may predict an explicit 'None' value for orders with no reordered items.
order_products_train.csv
order_id,product_id,add_to_cart_order,reordered 
Similar to the above in structure, difference being, that the above contains all prior orders before a particular order, which is present in this file. This should be used for training and validation.
order_products_test.csv
order_id,product_id,add_to_cart_order,reordered 
Similar to the prior data in structure, difference being, that it contains all prior orders before a particular order, which is present in this file. This should used for final testing after the model is decided.
orders.csv
order_id,user_id,eval_set,order_number,order_dow,order_hour_of_day,days_since_prior_order
This file tells to which set (prior, train, test) an order belongs. You are predicting reordered items only for the test set orders. Some variable definitions:
●	eval_set: which evaluation set this order belongs in (i.e. train, test or prior)
●	order_number: the order sequence number for this user (1 = first, n = nth)
●	order_dow: the day of the week the order was placed on
●	order_hour_of_day: the hour of the day the order was placed on
●	days_since_prior: days since the last order, capped at 30 (with NAs for order_number = 1)

products.csv
product_id,product_name,aisle_id,department_id
Contains details on the products.



Get dataset from here:
https://www.kaggle.com/philippsp/exploratory-analysis-instacart/data
