import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../data/beeradvocate_000.csv')
print("Shape of the dataset : ", df.shape)


print('Duplicate review count : ', df.duplicated(['review_text']).sum())
print('Not available review count : ', df['review_text'].isna().sum())
print('Not available beer style count : ', df['beer_style'].isna().sum())

print("Distinct beer styles : ", df['beer_style'].nunique())
df_beer_style = df['beer_style'].value_counts()
print("Number of values for each beer type : ")
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df_beer_style)

df_beer_style.plot.bar()
plt.xticks([])
plt.show()