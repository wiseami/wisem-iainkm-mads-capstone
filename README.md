# wisem-iainkm-mads-capstone







# Recommender Quick Start
Beyond the data analysis aspect of this repository, one of the key features is a music recommendation engine utilizing Spotify's API. Once launched, you're able to search for any artist, pick one of their top 5 songs and get back a recommended Daily Top 50 playlists from somewhere in the world that likely aligns with your choice.

## Spotify API Setup

Since portions of this code require a tie-in to Spotify's API, you'll need to create an account and register a basic app with them.
More information [here.](https://developer.spotify.com/documentation/general/guides/authorization/app-settings/)
Make sure when registering your app, you note down your CLIENT_ID and CLIENT_SECRET.

## Streamlit
### Installation
```
pip install streamlit
```

More complete instructions here at the [Streamlit repository.](https://github.com/streamlit/streamlit)

### Plugging in Spotify Information
Remember the CLIENT_ID and CLIENT_SECRET you were supposed to record? In order to utilize some of the live API functionality of the Streamlit app, plug those values into the ```secrets.toml``` file in the ```/.streamlit``` directory.
```
[spotify_credentials]
CLIENT_ID = "your_client_id_here"
CLIENT_SECRET = "your_client_secret_here"
```

Once you have Streamlit installed and your ```secrets.toml``` file setup, navigate to this root directory and start this up locally.
```
streamlit run streamlit_app.py
```





[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/wiseami/wisem-iainkm-mads-capstone/main/)