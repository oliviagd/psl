import dash
from dash import dcc, html, Input, Output, State, ALL
import pandas as pd
import numpy as np

sim_df = pd.read_csv("transformed_similarity_matrix.csv", index_col=0)

with open("ml-1m/movies.dat", 'r', encoding='latin1') as file:
    movies_raw = file.readlines()

movies = pd.DataFrame([line.strip().split("::") for line in movies_raw], columns=['MovieID', 'Title', 'Genres'])

movies['MovieID'] = movies['MovieID'].astype(int)
movies['MovieIDm'] = movies['MovieID'].apply(lambda x: f"m{x}")
movies['Year'] = movies['Title'].str.extract(r'\((\d{4})\)').astype(int)
movie_id_to_title_mapping = dict(zip(movies['MovieIDm'], movies['Title']))
movie_title_to_idmapping = dict(zip(movies['Title'], movies['MovieIDm']))

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Movie Recommendation System", style={"textAlign": "center"}),
    
    html.Div([
        html.H3("Rate the following movies:"),
        html.Div(id="movie-rating-inputs"),
        html.Button("Submit Ratings", id="submit-ratings", n_clicks=0),
        html.Div(id="error-message", style={"color": "red"})
    ]),
    
    html.Div([
        html.H3("Top 10 Recommendations:"),
        html.Ul(id="recommendations-list")
    ])
])

def IBCF(newuser, similarity_matrix):
    predicted_ratings = pd.Series(index=newuser.index)

    for i, idx in enumerate(newuser.index.values):
        if np.isnan(newuser[i]):
            S_i = similarity_matrix.iloc[i].dropna().index  # movies that are similar to movie i

            rated_indices = newuser.index[~newuser.isna()]
            common_indices = rated_indices.intersection(S_i)

            if len(common_indices) > 0:
                numerator = np.sum([similarity_matrix.iloc[i][j] * newuser[j] for j in common_indices])
                denominator = np.sum([similarity_matrix.iloc[i][j] for j in common_indices])

                predicted_rating = np.nan if denominator == 0 else numerator / denominator
                predicted_ratings.loc[idx] = predicted_rating
            else:
                predicted_ratings.loc[idx] = np.nan

    predicted_ratings.name = "pred"
    return predicted_ratings.sort_values(ascending=False)

def myIBCF(newuser, similarity_matrix):
    icbf_ranking = IBCF(newuser, similarity_matrix).head(10).dropna()

    popularity_ranking = pd.read_csv("movies_ranked_by_popularity.csv", usecols=['MovieIDm', 'WeightedScore'])
    popularity_ranking['ibcf'] = False
    popularity_ranking.columns = ['movie_id', 'rating', 'ibcf']

    icbf_ranking = icbf_ranking.reset_index()
    icbf_ranking['ibcf'] = True
    icbf_ranking.columns = ['movie_id', 'rating', 'ibcf']

    output = pd.concat([icbf_ranking, popularity_ranking]).drop_duplicates(subset=['movie_id'], keep='first').head(10)
    return output.head(10)

# Callback to dynamically populate movie rating inputs
@app.callback(
    Output("movie-rating-inputs", "children"),
    Input("submit-ratings", "n_clicks"),
    prevent_initial_call=False
)
def generate_rating_inputs(n_clicks):
    sample_movies = np.random.choice(list(movie_id_to_title_mapping.keys()), size=10, replace=False)
    inputs = [
        html.Div([
            html.Label(f"{movie_id_to_title_mapping[movie_id]}:"),
            dcc.Input(
                id={"type": "movie", "index": movie_id},
                type="number",
                placeholder="Rate 1-5 or leave blank",
                min=1,
                max=5,
                step=1,
                style={"marginBottom": "10px"}
            )
        ]) for movie_id in sample_movies
    ]
    return inputs

@app.callback(
    [Output("recommendations-list", "children"),
     Output("error-message", "children")],
    [Input("submit-ratings", "n_clicks")],
    [State({"type": "movie", "index": ALL}, "id"),
     State({"type": "movie", "index": ALL}, "value")],
    prevent_initial_call=True
)
def recommend_movies(n_clicks, input_ids, ratings):
    print(input_ids)
    print(ratings)
    if not ratings:
        return [], "Please rate at least one movie before submitting."
    
    user_ratings = pd.Series(data=np.nan, index=sim_df.index)
    input_ids = [ob['index'] for ob in input_ids]
    for movie_id, rating in zip(input_ids, ratings):
        if rating is not None and 1 <= rating <= 5:
            user_ratings.loc[movie_id] = rating
    # Ensure at least one movie is rated
    if user_ratings.notna().sum() == 0:
        return [], "Please rate at least one movie before submitting."
    
    # Run IBCF to get recommendations
    recommendations = myIBCF(user_ratings, sim_df)
    top_10_recommendations = recommendations.sort_values(by='rating',
                                ascending=False).head(10)['movie_id'].values
    # Convert to movie titles
    recommendation_titles = [movie_id_to_title_mapping.get(movie_id, "Unknown Title") for movie_id in top_10_recommendations]
    
    return [html.Li(title) for title in recommendation_titles], ""

if __name__ == "__main__":
    app.run_server(debug=True)

