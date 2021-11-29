import streamlit as st
import st_support_files.pages.st_home
import st_support_files.pages.st_density
import st_support_files.pages.st_correlations
import st_support_files.pages.st_kmeans
import st_support_files.pages.st_recommendation

### Start building out Streamlit assets
st.set_page_config(
    layout = "wide",
    menu_items = {'About':"Capstone project for University of Michigan's Master of Applied Data Science program by Mike Wise and Iain King-Moore"}
    )

### Define pages for navigation
PAGES = {
    "Home/Intro": st_support_files.pages.st_home,
    "Density Plots": st_support_files.pages.st_density,
    "Correlations": st_support_files.pages.st_correlations,
    "KMeans": st_support_files.pages.st_kmeans,
    "Recommendation": st_support_files.pages.st_recommendation
}

def write_page(page):
    """Writes the specified page/module
    Our multi-page app is structured into sub-files with a `def write()` function
    
    Arguments:
        page {module} -- A module with a 'def write():' function
    """
    page.write()

def main():
    """Main function of the App"""
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))

    page = PAGES[selection]

    with st.spinner(f"Loading {selection} ..."):
        write_page(page)
    st.sidebar.title("About")
    st.sidebar.info(
        """
        This app was built and maintained by Mike Wise and Iain King-Moore for a capstone project for University of Michigan's Master of Applied Data Science program.
        """
    )

if __name__ == "__main__":
    main()
