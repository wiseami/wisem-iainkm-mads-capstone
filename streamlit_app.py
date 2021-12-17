import streamlit as st
import st_support_files.pages.st_home
import st_support_files.pages.st_recommendation
import st_support_files.pages.st_density_choose
import st_support_files.pages.st_correlations_choose
import st_support_files.pages.st_country_clusters

### Start building out Streamlit assets
st.set_page_config(
    layout = "wide",
    menu_items = {'About':"Capstone project for University of Michigan's Master of Applied Data Science program by Mike Wise and Iain King-Moore"},
    page_title = 'Music Affinity Across Geographical Boundaries',
    initial_sidebar_state = 'expanded'
    )

### Define pages for navigation
PAGES = {
    "Home": st_support_files.pages.st_home,
    "Choose Your Own Density Plot": st_support_files.pages.st_density_choose,
    "Choose Your Own Correlations": st_support_files.pages.st_correlations_choose,
    "Clustering": st_support_files.pages.st_country_clusters,
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
    selection = st.sidebar.radio("Pages:", list(PAGES.keys()))

    page = PAGES[selection]

    #with st.spinner(f"Loading {selection} ..."):
    write_page(page)

    st.sidebar.title("About")
    st.sidebar.info(
        """
        This app was built and maintained by Mike Wise and Iain King-Moore for a capstone project for University of Michigan's Master of Applied Data Science program.

        Check out our [GitHub repo](https://github.com/wiseami/wisem-iainkm-mads-capstone).
        """
    )


if __name__ == "__main__":
    main()
