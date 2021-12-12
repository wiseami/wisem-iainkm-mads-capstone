# wisem-iainkm-mads-capstone

## Spotify API Setup

Since portions of this code require a tie-in to Spotify's API, you'll need to create an account and register a basic app with them.
More information [here.](https://developer.spotify.com/documentation/general/guides/authorization/app-settings/)
Make sure when registering your app, you note down your CLIENT_ID and CLIENT_SECRET.

## Streamlit

Remember the CLIENT_ID and CLIENT_SECRET you were supposed to record? In order to utilize some of the live API call functionality of the Streamlit app, plug those values into the ```secrets.toml``` file in the ```/.streamlit``` directory.
```
[spotify_credentials]
CLIENT_ID = "your_client_id_here"
CLIENT_SECRET = "your_client_secret_here"
```
