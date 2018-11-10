import pandas as pd
from src.main.config import data_path, processed_path

beerdf = pd.read_csv(data_path + '\\beeradvocate_000.csv')
beerdf = beerdf.drop_duplicates('review_text')
beerdf = beerdf.dropna(subset=['review_text'])
beerdf = beerdf.dropna(subset=['beer_style'])

beerdf = beerdf.groupby('beer_style').filter(lambda x: len(x) > 7000)
style_map = {
    'American IPA': 'Pale Ale',
    'American Double / Imperial IPA': 'Pale Ale',
    'American Double / Imperial Stout': 'Stout',
    'American Pale Ale (APA)': 'Pale Ale',
    'American Amber / Red Ale': 'Amber',
    'Russian Imperial Stout': 'Stout',
    'American Porter': 'Porter',
    'Belgian Strong Dark Ale': 'Belgian Ale',
    'Fruit / Vegetable Beer': 'Fruit / Vegetable Beer',
    'Witbier': 'Wheat Beer',
    'Tripel': 'Belgian Ale',
    'American Barleywine': 'Barleywine',
    'American Adjunct Lager': 'Pale Lager',
    'Belgian Strong Pale Ale': 'Belgian Ale',
    'Hefeweizen': 'Wheat Beer',
    'English Pale Ale': 'Pale Ale',
    'American Stout': 'Stout',
    'Saison / Farmhouse Ale': 'Pale Ale',
    'American Pale Wheat Ale': 'Wheat Beer',
    'American Strong Ale': 'Pale Ale',
    'Dubbel': 'Belgian Ale',
    'Märzen / Oktoberfest': 'Amber'
}
beerdf['beer_style'] = beerdf['beer_style'].map(style_map)
keep = ['beer_style', 'review_text']
beerdf[keep].to_csv(processed_path + '\\preprocessed.csv')

df_beer_style = beerdf['beer_style'].value_counts()
print("After mapping: ")
print(df_beer_style)

# sample 2000 records from each style of beer
test_beerdf = beerdf.groupby('beer_style').apply(lambda x: x.sample(2000)).reset_index(drop=True)
test_beerdf[keep].to_csv(processed_path + '\\small.csv')

# sample 9500 records ; min if we want to have equal nr of records from each type
train_beerdf = beerdf.groupby('beer_style').apply(lambda x: x.sample(9500)).reset_index(drop=True)
train_beerdf[keep].to_csv(processed_path + '\\train.csv')


