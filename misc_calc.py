import numpy as np
import pandas as pd
import scipy.stats as stats
import sympy as sym
from sympy.stats import Normal, density
from sympy.plotting import plot

# append module path when outside parent direcory
import sys
print(sys.path)
sys.path.append('/home/ajkappes/technical/py_funs') # if working module not in technical
import py_funs.test_funs as fun
# importing module we created in same parent directory

x = np.array([162, 200, 271, 320, 393, 508, 539, 629, 706, 777, 884, 1008, 1101, 1182, 1463,
              1603, 1984, 2355, 2880])
n = 19
z = 1.645
q = 1/n * np.sum(x - 150)

print('length = ', len(x))
print('summation is', q)
print('lower CI is', q / z)

cl = .8 - 1.645 * (np.sqrt(.8 * .2) / 50)
print('lower CL is', cl)

chi_crit95 = stats.chi2.ppf(.95, 2)
chi_crit05 = stats.chi2.ppf(.05, 2)
n, x = 2, 10
print('CI is (', round(n * x / chi_crit95, 3), round(n * x / chi_crit05, 3), ')')

print(2 * stats.chi2.ppf(.975, 10) / 1000)

fun.add(34, 56)
fun.c_to_f(45)

mu =  5
1- stats.poisson.cdf(2, 16.737)
q = stats.chi2.ppf(.995, 40) * 0.5
1 - stats.chi2.cdf(q, 20)

#### Classification Tree ####

# generate random data instead of loading from /home/ajkappes directory #
def vars(j, var):
    vars = []
    l = 1
    for i in range(j):
        vars.append(var + str(l))
        l += 1
    return vars

def generate_data(n, indicate, test_split):
    alpha = 0.75
    X = vars(2, 'x')
    df = pd.DataFrame({X[0]: stats.norm.rvs(loc=5, scale=0.75, size=n),
                       X[1]: stats.norm.rvs(loc=2, scale=0.25, size=n)})
    df['y'] = alpha * df[X[0]] + (1 - alpha) * df[X[1]]
    df['y_bin'] = np.where(df['y'] > df['y'].mean(), 1, 0)

    train_idx = list(np.random.choice(df.index.tolist(), size=int((1 - test_split) * len(df)), replace=False))
    if indicate == 'train':
        return df.loc[train_idx]

    else:
        return df.loc[[idx for idx in df.index if idx not in train_idx]]

df_train = generate_data(100,'train', 0.2)
X_train = df_train[['x1', 'x2']]
y_train = df_train['y']

df_test = generate_data(100, 'test', 0.2)
X_test = df_test[['x1', 'x2']]
y_test = df_test['y']

class node:
    def __init__(self, x, y, idx, min_leaf_sample):
        self.x = x
        self.y = y
        self.idx = idx
        self.min_leaf_sample = min_leaf_sample
        self.cost_init = np.inf

    def generate_split(self, x_j):
        x = self.x.values[idx, x_j]

        for i in range(x.shape[0]):
            split_left = x[x <= x[i]]
            split_right = x[x > x[i]]

            if len(split_left) < self.min_leaf_sample or len(split_right) < self.min_leaf_sample:
                continue

            cost = self.get_cost(left, right)
            if cost < self.cost_init:
                self.cost_init = cost
                self.x_j = x_j
                self.split = x[i]

    def get_cost(self, left, right):
        left_std = self.y[self.idx][left]
        right_std = self.y[self.idx][right]
        return left_std * len(split_left) + right_std * len(split_right)

    def generate_var_split(self):
        for j in range(self.x.shape[1]):
            if self.leaf == True:
                return
            x = self.split_var
            left = np.nonzero(x <= self.split)[0]
            right = np.nonzero(x > self.split)[0]
            self.left = node(self.x, self.y, self.idx[left], self.min_leaf_sample)
            self.right = node(self.x, self.y, self.idx[right], self.min_leaf_sample)

    @property
    def split_var(self):
        return self.x.values[self.idx, self.x_j]

    @property
    def leaf(self):
        return self.cost_init == float(np.inf)

    def predict_idx(self, x_i):
        if self.leaf == True:
            return np.mean(self.y[self.idx])

        if x_i[self.x_j] <= self.split:
            d_node = self.left
        else:
            d_node = self.right

        return d_node.predict_idx(x_i)

    def prediction(self, x):
        return np.array([self.predict_idx(x_i) for x_i in x])

class decision_tree:

    def fit(self, x, y, min_leaf_sample):
        self.decisiontree = node(x, y, np.arange(len(y)), min_leaf_sample)
        return self

    def prediction(self, x):
        return self.decisiontree.prediction(x.values)


d_test_tree = decision_tree().fit(X_test, y_test, min_leaf_sample=10)
d_test_tree_predict = d_test_tree.prediction(X_test)

print('Prediction values for test data are:', '/n', d_test_tree_predict)

###### multiple-input model test ###############

from scipy.special import gamma
import numpy as np
import scipy.stats as stats

def gamma_pdf(x, theta, k):
    g_pdf = theta ** (-k) / gamma(k) * x ** (k - 1) * np.exp(- x / theta)
    if x > 0 and theta > 0 and k > 0:
        return g_pdf
    else:
        raise ValueError('x, theta, and k must be greater than 0')

gamma_pdf(3, 4, 1)




