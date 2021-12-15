# Music Affinity Across Geographical Boundaries
Capstone project for University of Michigan's Master of Applied Data Science program by Mike Wise and Iain King-Moore utilizing data pulled using Spotify's API to understand audio feature comparisons across the world.


# Getting Started
## Clone the repo
Clone this repository to get started.
```
git clone https://github.com/wiseami/wisem-iainkm-mads-capstone.git
```

## Prereqs
Get all of the dependencies needed.
```
pip install -r requirements.txt
```

# Recommendation Engine Quick Start
Beyond the data analysis aspect of this repository, one of the key features is a music recommendation engine utilizing Spotify's API. Once launched, you're able to search for any artist or song and get back a recommended Daily Top 50 playlists from somewhere in the world that likely aligns with your choice.

## Spotify API Setup

Since portions of this code require a tie-in to Spotify's API, you'll need to create an account and register a basic app with them.
[Spotify](https://developer.spotify.com/documentation/general/guides/authorization/app-settings/) has a great step-by-step guide on creating a simple app needed to leverage their API.
\
\
Make sure when registering your app, you note down your CLIENT_ID and CLIENT_SECRET as you'll need those for a Streamlit configuration file.

## Streamlit
### Installation
This should be taken care of if you used the `requirements.txt` above, but in order for you to sucessfully run the Streamlit app locally, you'll need the Streamlit library. 
```
pip install streamlit
```

For more information about Streamlit in general, check out their [repository.](https://github.com/streamlit/streamlit)

### Plugging in Spotify Information
Remember the CLIENT_ID and CLIENT_SECRET you were supposed to record? In order to utilize some of the live API functionality of the Streamlit app, plug those values into the ```secrets.toml``` file in the ```/.streamlit``` directory.
```
[spotify_credentials]
CLIENT_ID = "your_client_id_here"
CLIENT_SECRET = "your_client_secret_here"
```

Once you have Streamlit installed and your ```secrets.toml``` file setup, navigate to this root directory and start it up locally.
```
streamlit run streamlit_app.py
```

Good work! Now you can navigate to [http://localhost:8501/](http://localhost:8501/) and see the app for yourself, assuming you didn't change the default port.

### Streamlit Cloud
Check out our already published app hosted in Streamlit Cloud!\
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/wiseami/wisem-iainkm-mads-capstone/main/)


# Data Flow
To get an idea of what the scripts in this repo are used for, below is a snapshot of the data flow. Some ran daily and some were either used once after our data collection was complete to find additional audio features or for analysis. 
![Data Flow Doc](https://github.com/wiseami/wisem-iainkm-mads-capstone/blob/main/assets/data_flow.png?raw=true)