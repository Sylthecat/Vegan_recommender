import streamlit as st
import streamlit.components.v1 as stc

import pandas as pd
import neattext.functions as nfx
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel


# Load the vegan recipes dataset
@st.cache_data
def load_data(data):
    dataset = pd.read_csv(data)
    dataset['Joined'] = dataset.apply(lambda x: f"{x['title']}{x['ingredients']}{x['preparation']}", axis=1)

    dataset['pp_Joined'] = dataset['Joined'].apply(nfx.remove_stopwords)
    dataset['pp_Joined'] = dataset['pp_Joined'].apply(nfx.remove_terms_in_bracket)
    dataset['pp_Joined'] = dataset['pp_Joined'].apply(nfx.remove_special_characters)
    dataset['pp_Joined'] = dataset['pp_Joined'].apply(nfx.remove_numbers)
    dataset['pp_Joined'] = dataset['pp_Joined'].apply(nfx.remove_shortwords)
    dataset['pp_Joined'] = dataset['pp_Joined'].str.lower()
    # data['pp_combined_features'] = data['pp_combined_features'].map(remover)
    dataset['pp_Joined'] = dataset['pp_Joined'].apply(nfx.remove_punctuations)
    dataset['title'] = dataset['title'].str.lower()
    return dataset


# Vectorisation and Cosine Similarity Matrix

def vectorise_text_to_cm(data):
    cv = CountVectorizer()
    cvm = cv.fit_transform(data)
    # Get the cosine
    csm = cosine_similarity(cvm)
    return csm


@st.cache_resource
def get_recommendation(joined, csm, dataset, no_rec=10):
    # indices of the recipe
    indices = pd.Series(dataset.index, index=dataset['title']).drop_duplicates()
    # Index of the recipe
    idx = indices[joined]
    # find the index with cosine matrix
    similarity = list(enumerate(csm[idx]))
    similarity = sorted(similarity, key=lambda x: x[1], reverse=True)
    selected_recipe_indices = [i[0] for i in similarity[1:]]
    selected_recipe_similarity = [i[1] for i in similarity[1:]]

    # Get the dataset and recipe title
    result_dataset = dataset.iloc[selected_recipe_indices]
    result_dataset['similarity'] = selected_recipe_similarity
    final_recommendations = result_dataset[['title', 'similarity', 'href', 'ingredients', 'preparation']]
    return final_recommendations.head(no_rec)


RESULT_TEMP = """
<div style="width:90%;height:100%;margin:1px;padding:5px;position:relative;border-radius:5px;border-bottom-right-radius: 80px;
box-shadow:0 0 15px 5px #ccc; background-color: #b3b3cc;
  border-left: 5px solid #6c6c6c;">
<h4>{}</h4>
<p style="color:black;"><span style="color:black;">Similarity:</span>{}</p>
<p style="color:Blue;"><span style="color:black;">🔗</span><a target="_blank" href="{}">Link</a></p>
<p style="color:black;">{}</p>
<p style="color:black;"><span style="color:black;">Preparation:</span>{}</p>
</div>
"""


# Search For Recipe
def search_term_if_not_found(term, dataset):
    result_dataset = dataset[dataset['pp_Joined'].str.contains(term)]
    return result_dataset


def main():
    st.title('Recipe Recommendation App')

    menu = ["Home", "Recommend", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    dataset = load_data("data/vegan_recipes.csv")

    if choice == "Home":
        st.subheader("Home")
        st.dataframe(dataset.head(10))

    elif choice == "Recommend":
        st.subheader("Recommend Recipes")
        csm = vectorise_text_to_cm(dataset['pp_Joined'])
        search_term = st.text_input("search").lower()
        print(search_term)
        no_rec = st.sidebar.number_input("Number", 4, 30, 7)
        if st.button("Recommend"):
            if search_term is not None:
                try:
                    results = get_recommendation(search_term, csm, dataset, no_rec)
                    with st.expander("Results as JSON"):
                        results_json = results.to_dict('index')
                        st.write(results_json)

                    for row in results.iterrows():
                        rec_title = row[1][0]
                        rec_score = row[1][1]
                        rec_link = row[1][2]
                        rec_ingredients = row[1][3]
                        rec_preparation = row[1][4]

                        stc.html(RESULT_TEMP.format(rec_title, rec_score, rec_link, rec_ingredients, rec_preparation),
                                 height=350)

                except:
                    results = "Not Found"
                    st.warning(results)
                    st.info("Suggested Options include")
                    results_dataset = search_term_if_not_found(search_term, dataset)
                    st.dataframe(results_dataset)


        else:
            st.subheader("About")
            st.text("Built with Streamlit & Pandas")


if __name__ == '__main__':
    main()
