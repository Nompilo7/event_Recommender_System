import streamlit as st
import dill as pickle
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit.components.v1 as components

# Load the vectorizer and recommendation function
with open('vec13.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
    
with open('recommendations_func13.pkl', 'rb') as f:
    get_recommendations_content = pickle.load(f)

# Load your dataset
df = pd.read_csv('aa_data.csv', index_col=0)

# Initialize session state variables
if 'view_event' not in st.session_state:
    st.session_state.view_event = None
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'filters' not in st.session_state:
    st.session_state.filters = {}
if 'page' not in st.session_state:
    st.session_state.page = 'recommendations'
if 'original_recommendations' not in st.session_state:
    st.session_state.original_recommendations = None

# Function to render OpenStreetMap static map
def render_map(lat, lon):
    map_url = f"https://www.openstreetmap.org/export/embed.html?bbox={lon-0.01},{lat-0.01},{lon+0.01},{lat+0.01}&layer=mapnik"
    iframe = f"""
    <iframe
        width="100%"
        height="400"
        frameborder="0"
        scrolling="no"
        marginheight="0"
        marginwidth="0"
        src="{map_url}">
    </iframe>
    """
    components.html(iframe, height=400)


def show_recommendations_page():
    st.title('Event Recommendation System')

    # Input content and control buttons
    input_content = st.text_input('🔍 Enter event description:', '')

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button('Get Recommendations'):
            if input_content.strip():  # Check if the input is not empty or just spaces
                # Get recommendations from the model
                recommendations = get_recommendations_content(input_content, df, vectorizer)
                
                # Merge recommendations with the original dataframe to get additional details
                recommended_events = df[df['Event_id'].isin(recommendations['Event_id'])]

                # Save the original recommendations
                st.session_state.original_recommendations = recommended_events.copy()

                # Limit the number of displayed events to 20
                st.session_state.recommendations = recommended_events.head(20)

    with col2:
        if st.button('Apply Filters'):
            if st.session_state.recommendations is not None:
                st.session_state.page = 'filtering'
                st.experimental_rerun()
            else:
                st.warning("Please get recommendations first before applying filters.")

    # Show filter options if recommendations are available
    if st.session_state.recommendations is not None:
        st.write("### Filter Options")

        # Display event details
        num_cols = 3
        cols = st.columns(num_cols)
        
        for idx in range(len(st.session_state.recommendations)):
            col = cols[idx % num_cols]
            with col:
                st.image(st.session_state.recommendations["Image_link"].iloc[idx], width=150)
                st.markdown(f"<h3 style='font-size: 18px;'>{st.session_state.recommendations['Event_name'].iloc[idx]}</h3>", unsafe_allow_html=True)
                
                if st.button(f"View", key=f"view_btn_{idx}"):  # No longer display numbers like "View 0"
                    st.session_state.view_event = st.session_state.recommendations.iloc[idx].to_dict()
                    st.session_state.page = 'event_details'
                    st.experimental_rerun()

                st.markdown("---")

    else:
        st.warning("Please enter a description to get recommendations.")
        
def show_filtering_page():
    if st.session_state.original_recommendations is None:
        st.warning("Please get recommendations first before applying filters.")
        st.button("Back to Recommender", key="back_to_recommender", on_click=lambda: st.session_state.update({'page': 'recommendations', 'recommendations': None}))
        return

    st.sidebar.title('Filters')

    # Show filter options
    location_province_filter = st.sidebar.multiselect('Filter by Location Province', df['Location_Province'].unique())
    event_type_filter = st.sidebar.multiselect('Filter by Event Type', df['Event_type'].unique())
    day_filter = st.sidebar.multiselect('Filter by Day', df['day'].unique())
    time_period_filter = st.sidebar.multiselect('Filter by Time Period', df['time_period'].unique())
    day_of_week_filter = st.sidebar.multiselect('Filter by Day of the Week', df['day_of_the_week'].unique())

    # Add sort by price option
    sort_by_price = st.sidebar.selectbox('Sort by Price', ['None', 'Ascending', 'Descending', 'Free Events'])

    # Apply filters independently
    filtered_df = st.session_state.original_recommendations.copy()

    if location_province_filter:
        filtered_df = filtered_df[filtered_df['Location_Province'].isin(location_province_filter)]
    if event_type_filter:
        filtered_df = filtered_df[filtered_df['Event_type'].isin(event_type_filter)]
    if day_filter:
        filtered_df = filtered_df[filtered_df['day'].isin(day_filter)]
    if time_period_filter:
        filtered_df = filtered_df[filtered_df['time_period'].isin(time_period_filter)]
    if day_of_week_filter:
        filtered_df = filtered_df[filtered_df['day_of_the_week'].isin(day_of_week_filter)]

    # Apply sorting by price or filter for free events
    if sort_by_price == 'Ascending':
        filtered_df = filtered_df.sort_values(by='Price', ascending=True)
    elif sort_by_price == 'Descending':
        filtered_df = filtered_df.sort_values(by='Price', ascending=False)
    elif sort_by_price == 'Free Events':
        filtered_df = filtered_df[filtered_df['Price'] == 0]

    # Update recommendations with filtered and sorted data
    st.session_state.recommendations = filtered_df

    # Display event details
    st.title('Filtered Event Recommendations')
    num_cols = 3
    cols = st.columns(num_cols)
    
    for idx in range(len(st.session_state.recommendations)):
        col = cols[idx % num_cols]
        with col:
            st.image(st.session_state.recommendations["Image_link"].iloc[idx], width=150)
            st.markdown(f"<h3 style='font-size: 18px;'>{st.session_state.recommendations['Event_name'].iloc[idx]}</h3>", unsafe_allow_html=True)
            
            if st.button(f"View", key=f"view_btn_{idx}"):  # No longer display numbers like "View 0"
                st.session_state.view_event = st.session_state.recommendations.iloc[idx].to_dict()
                st.session_state.page = 'event_details'
                st.experimental_rerun()

            st.markdown("---")

    # Add a "Back to Recommender" button to return to the recommender page
    if st.button("Back to Recommender"):
        st.session_state.page = 'recommendations'
        st.session_state.recommendations = None  # Reset recommendations to allow for new searches
        st.experimental_rerun()



def show_event_details_page():
    if st.session_state.view_event is not None:
        # Display detailed information about the selected event
        st.write("## Event Details")
        event = st.session_state.view_event
        
        # Display the event image and title
        st.image(event.get('Image_link', ''), width=300)
        st.write(f"### {event.get('Event_name', 'No Name')}")
        
        # Format price with Rand symbol
        price = event.get('Price', None)
        if price is not None:
            formatted_price = f"R {price:,.2f}"
            st.write(f"**Price:** {formatted_price}")
        
        # Helper function to remove latitude and longitude from the address
        def remove_lat_lon(address):
            # Use regular expression to remove the ' - Lat: ... , Lon: ...' part
            return re.sub(r' - Lat: [\-\d.]+, Lon: [\-\d.]+', '', address)
        
        # Capitalize column names for display
        for label, value in event.items():
            # Capitalize the label (column name) and exclude certain fields
            if label not in ['Longitude', 'Latitude', 'index', 'Event_id', 'Unnamed: 0:', 'Image_link', 'Event_name', 'Location_Province', 'Month', 'Content', 'Price','time_period','day_of_the_week','day']:
                capitalized_label = label.replace('_', ' ').title()  # Capitalize and format label
                # If the label is 'Address', clean the address
                if label == 'Address':
                    value = remove_lat_lon(value)
                st.write(f"**{capitalized_label}:** {value}")

        # Map and Directions
        if 'Latitude' in event and 'Longitude' in event:
            lat = event['Latitude']
            lon = event['Longitude']
            st.write("### Location")
            # Render OpenStreetMap static map
            render_map(lat, lon)
            # Display link for directions
            directions_url = f"https://www.openstreetmap.org/directions?from=&to={lat},{lon}"
            st.write(f"[Get Directions]({directions_url})")

        # Option to go back to recommendations
        if st.button("Back to Recommendations", key="back_to_recommendations"):
            st.session_state.view_event = None
            st.session_state.page = 'recommendations'
            st.experimental_rerun()


# Main Page Routing
if st.session_state.page == 'recommendations':
    show_recommendations_page()
elif st.session_state.page == 'filtering':
    show_filtering_page()
elif st.session_state.page == 'event_details':
    show_event_details_page()