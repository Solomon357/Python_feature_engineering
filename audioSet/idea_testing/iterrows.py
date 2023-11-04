import pandas as pd

df = pd.DataFrame({'c1': [10, 30], 'c2': [100, 200], 'c3': [20, 400]})
df = df.reset_index()  # make sure indexes pair with number of rows

# for ix, row in df.iterrows():
#     #print(row['c1'], row['c2'])
#     df.iloc[:,:-4]

print(df['c1'])
print(df['c1'].shape)



data = {'Products': ['Computer', 'Printer', 'Tablet', 'Chair', 'Desk'],
        'Brand':['A', 'B', 'C', 'D', 'E'],
        'Price':[750, 200, 300, 150, 400]
        }
df_a = pd.DataFrame(data, columns = ['Products', 'Brand', 'Price'])

## acc important ** a row is automatically considered a series
my_series = df_a.iloc[3]
print("a row being turned into a series")
print(my_series)
print(type(my_series))
print(my_series.shape)


