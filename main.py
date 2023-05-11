import pandas as pd
import neattext.functions as nfx
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('data/vegan_recipes.csv')

# print(data.head())

# print(data['ingredients'])


# Preprocessing: stopwords, special characters
# cust_set = set(non_food_ingredients)


# def remover(p):
#   return ' '.join([x for x in p.split() if x not in cust_set])
join_columns = lambda x: f"{x['title']}{x['ingredients']}{x['preparation']}"
data['Joined'] = data.apply(join_columns, axis=1)

data['pp_Joined'] = data['Joined'].apply(nfx.remove_stopwords)
data['pp_Joined'] = data['pp_Joined'].apply(nfx.remove_terms_in_bracket)
data['pp_Joined'] = data['pp_Joined'].apply(nfx.remove_special_characters)
data['pp_Joined'] = data['pp_Joined'].apply(nfx.remove_numbers)
data['pp_Joined'] = data['pp_Joined'].apply(nfx.remove_shortwords)
data['pp_Joined'] = data['pp_Joined'].str.lower()
# data['pp_combined_features'] = data['pp_combined_features'].map(remover)
data['pp_Joined'] = data['pp_Joined'].apply(nfx.remove_punctuations)

# print(data[['pp_ingredients', 'ingredients']])

# vecorize the ingredients
count_vect = CountVectorizer()
cv_mat = count_vect.fit_transform(data['pp_Joined'])
# sparse
# print (cv_mat)

# print(cv_mat.shape)
# Dense
data_cv_words = pd.DataFrame(cv_mat.todense(), columns=count_vect.get_feature_names_out())

# print(*data_cv_words)

# Cosine similarity
cosine_sim_mat = cosine_similarity(cv_mat)

# print (cosine_sim_mat)
sns.heatmap(cosine_sim_mat, yticklabels=False, annot=False)

# plt.show()

indices = pd.Series(data.index, index=data['title']).drop_duplicates()

print(indices)

def recommend_recipe(joined, num_of_rec=10):
    print(joined)
    idx = indices[joined]
    # Course Indices
    # Search inside cosine_sim_mat
    print(idx)
    scores = list(enumerate(cosine_sim_mat[idx]))
    # Scores
    # Sort Scores
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # Recommendation
    selected_indices = [i[0] for i in sorted_scores[1:]]
    selected_scores = [i[1] for i in sorted_scores[1:]]

    rec_recipes = data['title'].iloc[selected_indices]
    rec_data = pd.DataFrame(rec_recipes)
    rec_data['similarity_scores'] = selected_scores
    return rec_data.head(num_of_rec)


print(
    recommend_recipe('Almond Flour Crackers'))

data.to_csv("data.vegan_recipes_clean.csv")
