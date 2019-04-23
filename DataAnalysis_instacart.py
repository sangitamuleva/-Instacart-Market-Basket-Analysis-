
import warnings

warnings.filterwarnings('ignore')
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as stats

color = sns.color_palette()

#get_ipython().run_line_magic('matplotlib', 'inline')
# sets the backend of matplotlib as inline backend.

#create dataframes and load all the data files
orders_df = pd.read_csv(r'C:\Users\Admin\Contacts\Desktop\sangita\Project and Data Sets\Data\orders.csv').drop(['Unnamed: 0'], axis=1)
prior_df = pd.read_csv(r'C:\Users\Admin\Contacts\Desktop\sangita\Project and Data Sets\Data\order_products_prior.csv')
train_df = pd.read_csv(r'C:\Users\Admin\Contacts\Desktop\sangita\Project and Data Sets\Data\order_products_train.csv').drop(['Unnamed: 0'], axis=1)
products_df = pd.read_csv(r'C:\Users\Admin\Contacts\Desktop\sangita\Project and Data Sets\Data\products.csv')
aisles_df = pd.read_csv(r'C:\Users\Admin\Contacts\Desktop\sangita\Project and Data Sets\Data\aisles.csv',index_col=0)
departments_df = pd.read_csv(r'C:\Users\Admin\Contacts\Desktop\sangita\Project and Data Sets\Data\departments.csv')
test_df = pd.read_csv(r'C:\Users\Admin\Contacts\Desktop\sangita\Project and Data Sets\Data\order_products_test.csv').drop(['Unnamed: 0'], axis=1)

#count all the value of each eval_set
orders_df['eval_set'].value_counts()


unique_users = orders_df.groupby("eval_set")["user_id"].nunique()
unique_users

# nunique return count
# 131209 unique users
# Last order of each user is divided into train and test.
# Training contains 101209 orders and testing contains 30000 orders.

orders_df[''].boxplot()
# In[9]:


orders_per_user = orders_df.groupby("user_id")["order_number"].max().reset_index()
orders_per_user
orders_per_user = orders_per_user['order_number'].value_counts()
orders_per_user

plt.figure(figsize=(12, 6))
sns.barplot(orders_per_user.index, orders_per_user.values, alpha=0.8, color='blue')
plt.ylabel('Number of Customers', fontsize=12)
plt.xlabel('Number of Orders ', fontsize=12)
plt.title('Prior Orders Per Customer', fontsize=15)
plt.xticks(rotation='vertical')
plt.show()

# Around 15300 users made order for 4 times
# Decrease in the number of customers as the number of order increases
# With maximum orders capped to 100.


# In[84]:


products_per_order = prior_df.groupby("order_id")["add_to_cart_order"].max().reset_index()
products_per_order
products_per_order = products_per_order['add_to_cart_order'].value_counts()

plt.figure(figsize=(8, 6))
sns.barplot(products_per_order.index, products_per_order.values, color='orange')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Products', fontsize=12)
plt.title('Number of Products Per Order', fontsize=15)
plt.xlim(xmax=50)
plt.xticks(rotation='vertical')
plt.show()


# Max Products per customer order is 5
# MEdian - 8


# In[54]:


def make_day(x):
    return {
        0: 'Saturday',
        1: 'Sunday',
        2: 'Monday',
        3: 'Tuesday',
        4: 'Wednesday',
        5: 'Thursday',
    }.get(x, 'Friday')


# In[55]:


orders_df['order_dow'] = orders_df['order_dow'].map(make_day)
dow = orders_df.groupby('order_dow')[['order_id']].count().reset_index().sort_values('order_id', ascending=False)
plt.figure(figsize=(7, 7))
sns.barplot(x='order_dow', y='order_id', data=dow, color='green')
plt.ylabel('Total Orders', fontsize=12)
plt.xlabel('Day of week', fontsize=12)
plt.title("Total Orders by Day of Week", fontsize=15)
plt.show()

# Considering
# 0 - Saturday
# 1 - Sunday
# Max orders are made on Saturday and Sunday


# In[56]:


# No of Orders by Hour

plt.figure(figsize=(8, 6))
sns.countplot(x="order_hour_of_day", data=orders_df, color='blue')
plt.ylabel('Number of Orders', fontsize=12)
plt.xlabel('Hour of Day', fontsize=12)
plt.title("Number of Orders by Hour", fontsize=15)
plt.show()

# Max orders are made during late morning (10 - 11)&
# In the afternoon from (1 - 4)


# In[85]:


# Total Orders by Days Since Prior Order

sample = orders_df[orders_df['days_since_prior_order'].notnull()]
sample['days_since_prior_order'] = sample['days_since_prior_order'].map(lambda x: int(x))
plt.figure(figsize=(7, 7))
sns.countplot(x="days_since_prior_order", data=sample, color='purple')
plt.ylabel('Total Orders', fontsize=12)
plt.xlabel('Days Since Prior order', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("Total Orders by Days Since Prior Order", fontsize=15)
plt.show()

# In[86]:


duplicates = prior_df.groupby(['order_id', 'product_id'])[['product_id']].count()
duplicates.columns = ['count']
duplicates = duplicates.reset_index()
print('Number of instances of an item having a quanity greater than one in an order: ' +
      str(len(duplicates[duplicates['count'] > 1])))

# In[87]:


# Most Ordered Products

opp = pd.merge(prior_df, products_df, on='product_id', how='inner')
opp = pd.merge(opp, departments_df, on='department_id', how='inner')
opp = pd.merge(opp, aisles_df, on='aisle_id', how='inner')

# In[88]:


dept_freq = opp['department'].value_counts().head(5)
plt.figure(figsize=(8, 6))
ax = sns.barplot(dept_freq.index, dept_freq.values, color='red')
plt.ticklabel_format(style='plain', axis='y', scilimits=(0, 0))
plt.ylabel('Products Sold', fontsize=12)
plt.xlabel('Departments', fontsize=12)
plt.title('Products Sold By Highest Volume Departments', fontsize=15)
plt.show()

# In[89]:


aisle_freq = opp['aisle'].value_counts().head(5)
plt.figure(figsize=(8, 6))
sns.barplot(aisle_freq.index, aisle_freq.values, alpha=0.8)
plt.ylabel('Number of Orders', fontsize=12)
plt.xlabel('Aisle', fontsize=12)
plt.title('Products Sold By Highest Volume Aisles', fontsize=15)
plt.xticks(rotation='vertical')
plt.show()

# In[90]:


opp_orders = pd.merge(opp, orders_df, on='order_id', how='inner')
prod_orders = opp_orders.groupby('product_id')[['order_id']].count().reset_index()
prod_orders.columns = ['product_id', 'prod_orders']
prod_orders = pd.merge(prod_orders, products_df, on='product_id', how='inner')
prod_orders = pd.merge(prod_orders, departments_df, on='department_id', how='inner')
prod_orders.head(1)

# In[91]:


plt.figure(figsize=(8, 8))
dept_list = ['dairy eggs', 'snacks', 'beverages', 'frozen', 'pantry', 'bakery', 'produce']
mask = prod_orders['department'].isin(dept_list)
ax = sns.stripplot(x="department", y="prod_orders", data=prod_orders[mask], jitter=True)
ax.set(xlabel='Department', ylabel='Product Orders', title='Product Orders By Department')
plt.ylim(ymin=0)
plt.ylim(ymax=50000)
plt.show()

# In[92]:


most_ordered = prod_orders[['product_name', 'prod_orders']].sort_values('prod_orders', ascending=False).head(15)
ax = sns.barplot(y='product_name', x='prod_orders', data=most_ordered, color='crimson')
ax.set(xlabel='Total Orders', ylabel='Products', title='Most Ordered Products')
plt.show()

# In[93]:


print('Unique Products: ' + str(products_df['product_id'].nunique()))
print('Median Product Orders: ' + str(prod_orders['prod_orders'].median()))
print('Mean Product Orders: ' + str(prod_orders['prod_orders'].mean()))

# In[94]:


products_df['product_id'][49688] = 49699
products_df['product_name'][49688] = 'None'
products_df['aisle_id'][49688] = 100
products_df['department_id'][49688] = 21

# In[96]:


prior = pd.merge(prior_df, orders_df, on='order_id', how='inner')
no_reorders = prior[prior['order_number'] != 1].groupby('order_id')[['reordered']].sum().reset_index()
no_reorders = no_reorders[no_reorders['reordered'] == 0]['order_id'].unique()
prior_sub = prior[prior['order_id'].isin(no_reorders)].drop('add_to_cart_order', axis=1)
prior_sub['product_id'] = 49699
prior = pd.concat([prior.drop('add_to_cart_order', axis=1), prior_sub.drop_duplicates()], ignore_index=True)

# In[97]:


prod_by_user = prior.groupby(['user_id', 'product_id'])[['order_id']].count().reset_index()
prod_by_user.columns = ['user_id', 'product_id', 'orders']
single_orders = prod_by_user[prod_by_user['orders'] == 1].groupby('product_id')[['orders']].count().reset_index()
single_orders.columns = ['product_id', 'single']
multiple_orders = prod_by_user[prod_by_user['orders'] > 1].groupby('product_id')[['orders']].count().reset_index()
multiple_orders.columns = ['product_id', 'multiple']
prod_reorder = pd.merge(single_orders, multiple_orders, on='product_id', how='left')
prod_reorder = prod_reorder.fillna(value=0)
prod_reorder['reorder_rate'] = prod_reorder['multiple'] / (prod_reorder['single'] + prod_reorder['multiple'])
prod_reorder[prod_reorder['product_id'] == 49699]

# In[99]:


prods = prod_reorder['product_id'].unique()
print('Products single ordered at least once: ' + str(prod_reorder['product_id'].nunique()))
products_subset = products_df[-products_df['product_id'].isin(prods)]
products_subset['reorder_rate'] = 1
products_subset = products_subset[['product_id', 'reorder_rate']]
print('Products only ever reordered: ' + str(products_subset['product_id'].nunique()))

# In[100]:


prod_reorder = prod_reorder[['product_id', 'reorder_rate']].sort_values('reorder_rate', ascending=False)
prod_reorder = pd.concat([prod_reorder, products_subset], ignore_index=True)

# In[101]:


product_totals = pd.merge(prod_orders, prod_reorder, on='product_id', how='inner')
mask = product_totals['prod_orders'] >= 1000
head = product_totals[mask].sort_values('reorder_rate', ascending=False).head(7)
tail = product_totals[mask].sort_values('reorder_rate').head(7)

# In[102]:


ax = sns.barplot(y='product_name', x='reorder_rate', color='green', data=head)
ax.set(xlabel='reorder rate', ylabel='products', title='Most Reordered Products')
plt.show()

# In[103]:


ax = sns.barplot(y='product_name', x='reorder_rate', hue='department', data=tail)
ax.set(xlabel='reorder rate', ylabel='products', title='Least Reordered Products')
plt.show()

# In[104]:


print('Median reorder rate: ' + str(prod_reorder['reorder_rate'].median()))
print('Mean reorder rate: ' + str(prod_reorder['reorder_rate'].mean()))

# In[106]:


grouped_df = train_df.groupby("order_id")["reordered"].sum().reset_index()
print('Percent of orders with no reordered products in training orders: ' +
      str(float(grouped_df[grouped_df['reordered'] == 0].shape[0]) / grouped_df.shape[0]))

# In[107]:


train_orders = orders_df[orders_df['eval_set'] == 'train']
train_days = train_orders.groupby('days_since_prior_order')[['order_id']].count().reset_index()
train_days.columns = ['days_since_prior_order', 'train_orders']
nulls = pd.merge(orders_df, grouped_df[grouped_df['reordered'] == 0], on='order_id', how='inner')
none_df = nulls.groupby('days_since_prior_order')[['order_id']].count().reset_index()
none_df.columns = ['days_since_prior_order', 'none_orders']
none_df = pd.merge(none_df, train_days, on='days_since_prior_order', how='left')
none_df['proportion_of_nones'] = none_df['none_orders'] / none_df['train_orders']
none_df['days_since_prior_order'] = none_df['days_since_prior_order'].map(lambda x: int(x))
mask = (none_df['days_since_prior_order'] >= 9) & (none_df['days_since_prior_order'] <= 21)
none_df.loc[none_df[mask].days_since_prior_order, 'proportion_of_nones'] = none_df[mask]['proportion_of_nones'].median()
mask = (none_df['days_since_prior_order'] >= 22) & (none_df['days_since_prior_order'] <= 29)
none_df.loc[none_df[mask].days_since_prior_order, 'proportion_of_nones'] = none_df[mask]['proportion_of_nones'].median()
fig, ax = plt.subplots(figsize=(14, 8))
sns.pointplot(y='proportion_of_nones', x='days_since_prior_order', color='blue', data=none_df, ax=ax)
ax.set(xlabel='days since prior order', ylabel='proportion of none orders', title=
'Proportion of Orders With No Reordered Products')
plt.show()

# In[108]:


train_hour = train_orders.groupby('order_hour_of_day')[['order_id']].count().reset_index()
train_hour.columns = ['order_hour_of_day', 'train_orders']
none_hour = nulls.groupby('order_hour_of_day')[['order_id']].count().reset_index()
none_hour.columns = ['order_hour_of_day', 'none_orders']
none_hour = pd.merge(none_hour, train_hour, on='order_hour_of_day', how='left')
none_hour['proportion_of_nones'] = none_hour['none_orders'] / none_hour['train_orders']
fig, ax = plt.subplots(figsize=(14, 8))
sns.pointplot(y='proportion_of_nones', x='order_hour_of_day', color='blue', data=none_hour, ax=ax)
ax.set(xlabel='hour of the day', ylabel='proportion of none orders', title=
'Proportion of Orders With No Reordered Products')
plt.show()

# In[109]:


train_dow = train_orders.groupby('order_dow')[['order_id']].count().reset_index()
train_dow.columns = ['order_dow', 'train_orders']
none_dow = nulls.groupby('order_dow')[['order_id']].count().reset_index()
none_dow.columns = ['order_dow', 'none_orders']
none_dow = pd.merge(none_dow, train_dow, on='order_dow', how='left')
none_dow['proportion_of_nones'] = none_dow['none_orders'] / none_dow['train_orders']
none_dow = none_dow.sort_values('proportion_of_nones', ascending=False)
fig, ax = plt.subplots(figsize=(14, 8))
sns.pointplot(y='proportion_of_nones', x='order_dow', color='blue', data=none_dow, ax=ax)
ax.set(xlabel='day of the week', ylabel='proportion of none orders', title=
'Proportion of Orders With No Reordered Products')
plt.show()


# In[110]:


def plot_none(train_orders_df, first_median_range=None, x_max=None, second_median_range=None,
              days_filter=None, title_add_on=None, lowest_median_num=None):
    if days_filter == None:
        train_total = train_orders_df.groupby('order_number')[['order_id']].count().reset_index()
        none_total = nulls.groupby('order_number')[['order_id']].count().reset_index()
    elif type(days_filter) == int:
        mask = train_orders_df['days_since_prior_order'] == days_filter
        train_total = train_orders_df[mask].groupby('order_number')[['order_id']].count().reset_index()
        mask = nulls['days_since_prior_order'] == days_filter
        none_total = nulls[mask].groupby('order_number')[['order_id']].count().reset_index()
    else:
        mask = (train_orders['days_since_prior_order'] >= days_filter[0]) & (
                train_orders['days_since_prior_order'] <= days_filter[1])
        train_total = train_orders_df[mask].groupby('order_number')[['order_id']].count().reset_index()
        mask = (nulls['days_since_prior_order'] >= days_filter[0]) & (nulls['days_since_prior_order'] <= days_filter[1])
        none_total = nulls[mask].groupby('order_number')[['order_id']].count().reset_index()
    train_total.columns = ['order_number', 'train_orders']
    none_total.columns = ['order_number', 'none_orders']
    none_total = pd.merge(none_total, train_total, on='order_number', how='left')
    none_total['proportion_of_nones'] = none_total['none_orders'] / none_total['train_orders']
    order_numbers = none_total['order_number'].unique()
    absent_order_numbers = pd.DataFrame(columns=['order_number', 'proportion_of_nones'])
    if type(lowest_median_num) == int:
        lowest_median = none_total[none_total['order_number'] >= lowest_median_num]['proportion_of_nones'].median()
    else:
        mask = (none_total['order_number'] >= lowest_median_num[0]) & (
                    none_total['order_number'] <= lowest_median_num[1])
        lowest_median = none_total[mask]['proportion_of_nones'].median()
    for i in range(4, 101):
        if i not in order_numbers:
            absent_order_numbers.loc[i, 'order_number'] = i
            absent_order_numbers.loc[i, 'none_orders'] = np.nan
            absent_order_numbers.loc[i, 'train_orders'] = np.nan
            absent_order_numbers.loc[i, 'proportion_of_nones'] = lowest_median
    none_total = pd.concat([none_total, absent_order_numbers], ignore_index=True)
    mask = (none_total['order_number'] >= first_median_range[0]) & (none_total['order_number'] <= first_median_range[1])
    none_total.set_value(none_total[mask].index, 'proportion_of_nones',
                         none_total[mask]['proportion_of_nones'].median())
    if second_median_range != None:
        mask = (none_total['order_number'] >= second_median_range[0]) & (
                none_total['order_number'] <= second_median_range[1])
        none_total.set_value(none_total[mask].index, 'proportion_of_nones',
                             none_total[mask]['proportion_of_nones'].median())
    if type(lowest_median_num) == int:
        none_total.set_value(none_total.order_number >= lowest_median_num,
                             'proportion_of_nones', lowest_median).sort_values('order_number')
    else:
        none_total.set_value(none_total.order_number >= lowest_median_num[0],
                             'proportion_of_nones', lowest_median).sort_values('order_number')
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.pointplot(y='proportion_of_nones', x='order_number', color='blue', data=none_total, ax=ax)
    if type(title_add_on) == str:
        ax.set(xlabel='total orders', ylabel='proportion of none orders', title=
        'Proportion of Orders With No Reordered Products' + title_add_on)
    else:
        ax.set(xlabel='total orders', ylabel='proportion of none orders', title=
        'Proportion of Orders With No Reordered Products')
    plt.xlim(xmax=x_max)
    plt.show()


# In[111]:


plot_none(train_orders, first_median_range=[12, 26], x_max=40, lowest_median_num=40, second_median_range=[27, 39])

# In[112]:


plot_none(train_orders, first_median_range=[7, 10], x_max=30, second_median_range=[11, 24], days_filter=[0, 2],
          lowest_median_num=25, title_add_on=', 0-2 Days Since Prior Order')

# In[113]:


plot_none(train_orders, [16, 20], x_max=30, second_median_range=[21, 29], days_filter=[3, 29], lowest_median_num=30,
          title_add_on=', 3-29 Days Since Prior Order')

# In[115]:


plot_none(train_orders, first_median_range=[13, 21], x_max=30,
          days_filter=30, lowest_median_num=[22, 41], title_add_on=', 30 Days Since Prior Order')

# In[116]:


prod_by_user = prior.groupby(['user_id', 'product_id'])[['order_id']].count().reset_index()
prod_by_user.columns = ['user_id', 'product_id', 'num_ordered']
prod_by_user.head(1)

# In[117]:


max_orders = prior.groupby('user_id')[['order_number']].max().reset_index()
max_orders.columns = ['user_id', 'total_orders']
prod_by_user = pd.merge(prod_by_user, max_orders, on='user_id', how='left')
prod_by_user['order_rate'] = prod_by_user['num_ordered'] / prod_by_user['total_orders']
prod_by_user.head(1)

# In[118]:


last_order = prior.groupby(['user_id', 'product_id'])[['order_number']].max().reset_index()
last_order.columns = ['user_id', 'product_id', 'last_order']
prod_by_user = pd.merge(prod_by_user, last_order, on=['user_id', 'product_id'], how='left')
prod_by_user['orders_since'] = prod_by_user['total_orders'] - prod_by_user['last_order']
prod_by_user.sample(1)

# In[120]:


prod_reorder_subset = prod_reorder[['product_id', 'reorder_rate']]
prod_by_user = pd.merge(prod_by_user, prod_reorder_subset, on='product_id', how='inner')
prod_by_user[prod_by_user['user_id'] == 4]


# In[121]:


def recently_ordered(prior_df, num_from_last, features_df, new_col):
    order = prior_df.groupby('user_id')[['order_number']].max() - num_from_last
    order = order.reset_index()
    prior_subset = prior_df[['user_id', 'order_number', 'product_id']]
    order = pd.merge(order, prior_subset, on=['user_id', 'order_number'], how='inner')
    order = order[['user_id', 'product_id']]
    updated_df = pd.merge(features_df, order, on=['user_id', 'product_id'], how='left', indicator=True)
    updated_df['_merge'] = updated_df['_merge'].map(lambda x: 1 if x == 'both' else 0)
    updated_df = updated_df.rename(columns={'_merge': new_col})
    return updated_df


# In[ ]:


prod_by_user = recently_ordered(prior, 0, prod_by_user, 'in_last')
prod_by_user = recently_ordered(prior, 1, prod_by_user, 'in_2nd_last')
prod_by_user = recently_ordered(prior, 2, prod_by_user, 'in_3rd_last')
prod_by_user = recently_ordered(prior, 3, prod_by_user, 'in_4th_last')
prod_by_user = recently_ordered(prior, 4, prod_by_user, 'in_5th_last')
prod_by_user.sample(n=1)

# In[ ]:


mask = (orders_df['eval_set'] == 'train') | (orders_df['eval_set'] == 'test')
orders_subset = orders_df[mask].drop(['order_id', 'eval_set', 'order_hour_of_day'], axis=1)
orders_subset['order_number'] = orders_subset['order_number'].map(lambda x: x - 1)
orders_subset = orders_subset.rename(columns={'order_number': 'total_orders'})
model = pd.merge(prod_by_user, orders_subset, on=['user_id', 'total_orders'], how='left')
model.sample(n=5)





