import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Series structure skip because of its simple


# Create Dataframe
# data = np.random.randint(0, 30, (5, 6))
# print(data)
# df = pd.DataFrame(data= data, index =["a", "b", "c", "d", "e"], columns=["cl1", "cl2", "cl3", "cl4", "cl5", "cl6"])
# print(df)


# Element selection operation
# data = np.random.seed(101)
# print(data)
# df = pd.DataFrame(data=np.random.randn(6, 5),index="A B C D E F".split(), columns=["val1 val2 val3 val4 val5".split()])
# print(df)
# print(df["val3"])
# print(df["val3"]["B":"D"])
# variable = ["val2","val5"]
# print(df[variable])
# print(df[["val2","val5"]])
# print(df["B":"D"])
# print(df["E":"E"])
# print(df["E":"E"][["val2", "val5"]])


# top level element selection operation
# loc & iloc
# array = np.random.randint(0, 50, (10, 5))
# df = pd.DataFrame(data=array, index="A B C D E F G H I J".split(),
#                   columns="val1 val2 val3 val4 val5".split())
# print(df)
# print(df.loc["A":"D"])
# print(df.iloc[0:4])
# print(df.loc["C", "val4"])
# print(df.iloc[2, 3])
# print(df.loc["C":"G", "val3"])  # return pandas series structure
# print(df.loc["C":"G", ["val3"]])
# print(df.loc["C":"G", "val3": "val5"])
# print(df.iloc[2:7, 2])
# print(df.iloc[2:7, [2]])
# print(df.loc["C":"H", "val2":"val4"])
# print(df.iloc[2:8, 1: 4])
# print(df.iloc[2:8, 1: 4].loc["E":"F",["val3"]])


# conditional selection
# array = np.random.randint(0, 50, (10, 5))
# df = pd.DataFrame(data=array, index="A B C D E F G H I J".split(),
#                   columns="val1 val2 val3 val4 val5".split())
# print(df)
# print(df > 20)
# print(df[df > 20])
# print(df["val1"] < 20)
# print(df[df["val1"] < 20])
# print(df[df["val1"] < 20]["val2"])
# print(df[df["val1"] < 20][["val2"]])
# print(df[df["val1"] < 20][["val2", "val5"]])
# print(df[(df["val1"] > 20) & (df["val4"] < 10)])
# print(df[(df < 35) | (df["val5"] > 20)])
# print(df.loc[df.val2 > 25, ["val2", "val3", "val5"]])


# add column in dataframe
# array = np.random.randint(0, 50, (10, 5))
# df = pd.DataFrame(data=array, index="A B C D E F G H I J".split(),
#                   columns="val1 val2 val3 val4 val5".split())
# print(df)
# df["val1"] = df["val2"]
# print(df["val1"])
# print(df["val2"])
# print(df["val5"])
# df["val3"] = df["val4"]
# print(df["val3"])
# print(df["val4"])
# print(df)
# print(np.arange(20, 30))
# df["val5"] = np.arange(20, 30)
# df["val6"] = np.arange(30,40)
# print(df)


# remove raw and column << drop func, inplace = True for the change in dataframe
# array = np.random.randint(0, 50, (7, 4))
# df = pd.DataFrame(data=array, index="A B C D E F G".split(),
#                   columns="val1 val2 val3 val4".split())
# print(df)
# df.drop(["val3", "val4"], axis=1, inplace=True)
# print(df)
# df.drop(["C", "F"], axis=0, inplace=True)
# print(df)


# Null values in pandas dataframes
# data = {'val1': [2, 4, np.nan, 6, np.nan, 8, 10],
#         'val2': [123, np.nan, 456, np.nan, 789, 246, 357],
#         'val3': ['France', 'Greece', 'USA', 'Japan', 'Sweden', 'Norway', 'Turkey']}
# df = pd.DataFrame(data)
# print(df)
# print(df.isnull())
# print(df.isnull().sum())
# print(len(df))
# print(df.isnull().sum() / len(df)*100)
# print(df.notnull())
# print(df.notnull().sum())
# print(df.notnull().sum().sum())
# print(df["val1"].notnull())
# print(df[df["val1"].notnull()])
# print(df.isnull().any())
# print(df.isnull().any(axis=1))
# condition = df.isnull().any(axis=1)
# print(df[condition])
# print(df[~condition])
# print(df.notnull().all(axis=1))


# Dropping null values in pandas dataframe dropna func
# df = pd.DataFrame({'val1': [2, 4, np.nan, 6, np.nan, 8, 10], 'val2': [123, np.nan, 456, np.nan, 789, 246, 357], 'val3': [
#                   'France', 'Greece', 'USA', 'Japan', 'Sweden', 'Norway', 'Turkey']})
# print(df)
# print(df.dropna()) #drop whole raw by default axis = 0
# print(df.dropna(axis=1)) #drop whole column
# df['val4'] = np.nan
# print(df)
# print(df.dropna(how='all'))
# print(df.dropna(how='all', axis=1)) #compete empty column removed
# df.dropna(how='all', axis=1,inplace=True) #make change in original dataframe


# fill nan value in dataframe fillna
# df = pd.DataFrame({'val1': [2, 4, np.nan, 6, np.nan, 8, 10], 'val2': [123, np.nan, 456, np.nan, 789, 246, 357], 'val3': [
#                   'France', 'Greece', 'USA', 'Japan', 'Sweden', 'Norway', 'Turkey']})
# print(df)
# print(df.fillna(100000))
# print(df["val1"].fillna(100000))
# print(df.mean())
# print(df.fillna(df.mean()))
# print(df.fillna({'val1': 100000, 'val2': 200000}))
# print(df['val1'].fillna(df.val2.mean()))
# print(df['val3'][0::2])
# df['val3'][0::2] = np.nan
# print(df['val3'].fillna("Turkey"))
# print(df)
# print(df.fillna(method='ffill')) #fill nan with above value by default axis is 0 if we set to the one first column value fill in second column
# print(df.fillna(method='pad')) #same as ffill(above)
# print(df.fillna(method='bfill')) #fill as below value
# print(df.fillna(method='backfill')) #same as bfill(above)


# setting index in pandas dataframe
# df = pd.DataFrame(data=np.random.randn(6, 5), index='A B C D E F'.split(),
#                   columns='val1 val2 val3 val4 val5'.split())
# print(df)
# print(df.reset_index())
# df.reset_index(drop=True, inplace=True) # remove old index(drop) and add new one(inplace)
# print(df)
# example = 'TR F NL RUS AUS AZ'.split()
# print(example)
# df['new_index'] = example
# print(df)
# df.set_index('new_index', inplace=True)
# print(df)


# multi index in pandas dataframe
# inside = ['class A', 'class B', 'class C', 'class A', 'class B', 'class C']
# outside = ['school 1', 'school 1', 'school 1',
#            'school 2', 'school 2', 'school 2']
# zip(outside, inside)
# multiindex = list(zip(outside, inside))
# print(multiindex)
# hier_index = pd.MultiIndex.from_tuples(multiindex)
# print(hier_index)
# df = pd.DataFrame(data=np.random.randint(70, 100, size=(6, 2)), index=hier_index, columns=['1st_semester', '2nd_semester'])
# print(df)


# Element selection in multi indexd dataframes
# inside = ['class A', 'class B', 'class C', 'class A', 'class B', 'class C']
# outside = ['school 1', 'school 1', 'school 1',
#            'school 2', 'school 2', 'school 2']
# zip(outside, inside)
# multiindex = list(zip(outside, inside))
# print(multiindex)
# hier_index = pd.MultiIndex.from_tuples(multiindex)
# print(hier_index)
# df = pd.DataFrame(data=np.random.randint(70, 100, size=(6, 2)), index=hier_index, columns=['1st_semester', '2nd_semester'])
# print(df)
# print(df['1st_semester'])
# print(df[['1st_semester']]) # return dataframes
# print(df.loc['school 1'].loc['class B'])
# print(df.loc['school 1'].loc[['class B']])
# print(df.index)
# print(df.index.names)
# df.index.names = ['schools', 'classes']
# print(df)


# select element using the xs Func in multi indexed dataframes(xs func)
# inside = ['class A', 'class B', 'class C', 'class A', 'class B', 'class C']
# outside = ['school 1', 'school 1', 'school 1',
#            'school 2', 'school 2', 'school 2']
# zip(outside, inside)
# multiindex = list(zip(outside, inside))
# print(multiindex)
# hier_index = pd.MultiIndex.from_tuples(multiindex)
# print(hier_index)
# df = pd.DataFrame(data=np.random.randint(70, 100, size=(6, 2)),
#                   index=hier_index, columns=['1st_semester', '2nd_semester'])
# df.index.names = ['schools', 'classes']
# print(df)
# print(df.xs('school 2'))
# print(df.xs(('school 2', 'class A')))
# print(df.xs(('school 2', 'class A'), level=[0, 1])) # school at 0 level and classes at 1 level return dataframe(write level numerically)
# print(df.xs('class A', level='classes'))  # write level !numerically
# print(df.xs('class A', level = 1)) #write level numerically


# concatenating pandas dataframes
# df1 = pd.DataFrame({'X': ['X0', 'X1', 'X2', 'X3'],
#                     'Y': ['Y0', 'Y1', 'Y2', 'Y3'],
#                     'Z': ['Z0', 'Z1', 'Z2', 'Z3'],
#                     'T': ['T0', 'T1', 'T2', 'T3']})

# df2 = pd.DataFrame({'X': ['X4', 'X5', 'X6', 'X7'],
#                     'Y': ['Y4', 'Y5', 'Y6', 'Y7'],
#                     'Z': ['Z4', 'Z5', 'Z6', 'Z7'],
#                     'T': ['T4', 'T5', 'T6', 'T7']})

# df3 = pd.DataFrame({'X': ['X8', 'X9', 'X10', 'X11'],
#                     'Y': ['Y8', 'Y9', 'Y10', 'Y11'],
#                     'Z': ['Z8', 'Z9', 'Z10', 'Z11'],
#                     'T': ['T8', 'T9', 'T10', 'T11']})

# print(df1)
# print(df2)
# print(df3)
#
# print(pd.concat([df1, df2, df3]).reset_index(drop=True))
# print(pd.concat([df1, df2, df3], ignore_index=True))
# print(pd.concat([df1, df2, df3], axis=1))
# df2.columns = ['X', 'Y', 'Z', 'A']
# print(df2)
# print(df1)
# print(pd.concat([df1, df2], ignore_index=True))
# print(pd.concat([df1, df2],ignore_index=True, join = 'inner'))
#
# df1 = pd.DataFrame({'X': ['X0', 'X1', 'X2', 'X3'],
#                     'Y': ['Y0', 'Y1', 'Y2', 'Y3'],
#                     'Z': ['Z0', 'Z1', 'Z2', 'Z3'],
#                     'T': ['T0', 'T1', 'T2', 'T3']}, index=[0, 1, 2, 3])
#
# df2 = pd.DataFrame({'X': ['X4', 'X5', 'X6', 'X7'],
#                     'Y': ['Y4', 'Y5', 'Y6', 'Y7'],
#                     'Z': ['Z4', 'Z5', 'Z6', 'Z7'],
#                     'T': ['T4', 'T5', 'T6', 'T7']}, index=[5, 6, 7, 8])
#
# df3 = pd.DataFrame({'X': ['X8', 'X9', 'X10', 'X11'],
#                     'Y': ['Y8', 'Y9', 'Y10', 'Y11'],
#                     'Z': ['Z8', 'Z9', 'Z10', 'Z11'],
#                     'T': ['T8', 'T9', 'T10', 'T11']}, index=[9, 10, 11, 12])
#
# print(df1)
# print(df2)
# print(df3)
# print(pd.concat([df1, df2, df3], axis=1))


# merge pandas dataframes
# df1 = pd.DataFrame({'employee': ['Julia', 'Marie', 'Adam', 'Nicole'],
#                    'department': ['Data Science', 'Web Development', 'Data Science', 'Cyber SSecurity'],
#                     'Year': ['2005', '2008', '2011', '2002']})

# df2 = pd.DataFrame({'employee': ['Nicole', 'Adam', 'Julia', 'Marie'],
#                     'country': ['Canada', 'England', 'USA', 'Germany'],
#                     'salary': ['22000', '16000', '20000', '17500']})

# df3 = pd.DataFrame({'manager': ['Abraham', 'Joseph', 'Kayara'],
#                     'department': ['Web Development', 'Cyber Security', 'Data Science']})

# print(df1)
# print(df2)
# print(df3)
# print(pd.concat([df1, df2],axis=1)) # unordered
# print(pd.merge(df1, df2))
# print(pd.merge(df1, df2, on='employee'))
# print(pd.merge(df1, df2, on='employee', how='left'))
# print(pd.merge(df1, df2, on='employee', how='right'))
# df4 = pd.merge(df1, df2)
# print(df4)
# print(pd.merge(df4, df3, on='department', how='right'))
# df5 = pd.merge(df4, df3, on='department', how='right')
# df6 = pd.DataFrame({'department': ['Web Development', 'Web Development', 'Cyber Security', 'Cyber Secturity', 'Data Science', 'Data Science'],
#                     'prog_lang': ['HTML', 'CSS', 'C++', 'SQL', 'PYTHON', 'R']})
# print(df6)
# print(pd.merge(df5, df6))
# df3 = pd.merge(df1, df2)
#
# print(df1)
# print(df2)
# print(df3)
# df4 = pd.DataFrame({'employee': ['Julia', 'Alex', 'Adam', 'Kayra'],
#                    'department': ['Data Science', 'Web Development', 'Data Science', 'Cyber Security'],
#                     'university': ['harvard', 'hamburg', 'oxford', 'metu']})
# print(df4)
# print(pd.merge(df3, df4, on=['employee', 'department']))
# print(pd.merge(df3, df4, on=['employee', 'department'], how='outer'))
# print(pd.merge(df3, df4, on=['employee', 'department'], how='left'))
# print(pd.merge(df3, df4, on=['employee', 'department'], how='right'))
# df5 = pd.DataFrame({'member': ['Georgr', 'Marie', 'Nicole', 'Donald'],
#                     'family member': ['4', '5', '1', '3']})
# print(df5)
# print(pd.merge(df3, df5, left_on='employee', right_on='member', how='outer'))
# print(pd.merge(df3, df5, left_on='employee', right_on='member', how='left'))
# print(pd.merge(df3, df5, left_on='employee', right_on='member', how='right'))


# join func(work on index)
# df1 = pd.DataFrame({'X': ['X0', 'X1', 'X2'],
#                    'Y': ['Y0', 'Y1', 'Y2']},
#                    index=['A0', 'A1', 'A2'])

# df2 = pd.DataFrame({'Z': ['Z0', 'Z1', 'Z2'],
#                     'T': ['T0', 'T1', 'T2']},
#                    index=['A0', 'A2', 'A3'])

# print(df1)
# print(df2)
# print(df1.join(df2))
# print(df2.join(df1))
# print(df1.join(df2, how='inner'))
# print(df1.join(df2, how='outer'))
# print(df1.join(df2))
#
# df3 = pd.DataFrame({'key': ['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6'],
#                     'Y': ['Y0', 'Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6']})

# df4 = pd.DataFrame({'key': ['X0', 'X1', 'X2'],
#                     'Z': ['Z0', 'Z2', 'Z3']})
# print(df3)
# print(df4)
# print(df3.join(df4, lsuffix='left', rsuffix='right'))
# print(df3.set_index('key'))
# print(df3.set_index('key').join(df4.set_index('key')))
# print(pd.merge(df3, df4, how='outer'))


# function that can be applied on pandas dataframe
# loading, examine a dataset from the seaborn library
# df = sns.load_dataset('diamonds')
# print(df)
# print(df.head())
# print(df.tail())
# print(df.shape)
# print(df.info())


# aggregation func on dataset
# df = sns.load_dataset('diamonds')
# print(df.head())
# print(df.mean())
# print(df['price'].mean()) # mean is almost equal to median than distribution seems to be normal
# print(df.median()) # mean(price) > median(price) that means right skewed distribution of price
# print(df['cut'].count())
# print(df['price'].min())
# print(df['price'].max()) # difference between min and max is more min value is 326 ans max value is 18823 so distribution of price right skewed
# sns.displot(df['price'])
# sns.displot(df['table'])
# print(df.std())
# print(df['price'].var())
# print(df['carat'].sum())
# print(df.describe())
# print(df.describe().transpose())
# print(df.describe().T)
# plt.show()


# Examine dataset 2
# df = sns.load_dataset('planets')
# print(df.head())
# print(df.tail())
# print(df.info())
# df['method'] = pd.Categorical(df['method'])
# print(df.info())


# grouping and aggregation function
# df = sns.load_dataset('planets')
# print(df.head())
# print(df['method'].value_counts())
# print(df['method'].unique())
# print(df['distance'].value_counts())
# print(df['distance'].value_counts(dropna=False))
# print(df.distance.isnull().sum())
# print(df.head())
# print(df.groupby('method'))
# print(df.groupby('method').mean())
# print(df.groupby('method')['distance'])
# print(df.groupby('method')[['distance']].mean())
# print(df.groupby('year')[['number']].sum())
# print(df[df['year'] == 1992])
# print(df.describe())
