import streamlit as st
import pickle
import pandas as pd
import numpy as np

# ================================
# PAGE CONFIGURATION
# ================================
st.set_page_config(
    page_title="Food Recommender",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================================
# CUSTOM CSS - DARK THEME WITH RED/YELLOW
# ================================
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0a0a0a;
    }
    
    /* Remove top padding */
    .block-container {
        padding-top: 2rem;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 700;
    }
    
    /* Main title with gradient */
    .main-title {
        font-size: 56px;
        font-weight: 900;
        background: linear-gradient(135deg, #ff4444 0%, #ffaa00 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 10px;
        padding: 20px 0;
    }
    
    .subtitle {
        color: #cccccc;
        text-align: center;
        font-size: 20px;
        margin-bottom: 60px;
    }
    
    /* Section headers */
    .section-header {
        font-size: 36px;
        font-weight: 800;
        color: #ffffff;
        margin: 60px 0 30px 0;
        padding-bottom: 15px;
        border-bottom: 3px solid;
        border-image: linear-gradient(135deg, #ff4444 0%, #ffaa00 100%) 1;
    }
    
    /* Category headers */
    .category-header {
        background: linear-gradient(135deg, #ff4444 0%, #ffaa00 100%);
        padding: 20px 30px;
        border-radius: 12px;
        color: white;
        font-size: 28px;
        font-weight: 700;
        margin: 40px 0 25px 0;
        box-shadow: 0 6px 20px rgba(255, 68, 68, 0.4);
        text-align: center;
    }
    
    /* Food cards */
    .food-card {
        background: linear-gradient(145deg, #1a1a1a 0%, #0f0f0f 100%);
        border-radius: 15px;
        padding: 25px;
        margin: 15px 0;
        border: 2px solid #2a2a2a;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5);
        height: 100%;
    }
    
    .food-card:hover {
        border-color: #ff4444;
        box-shadow: 0 8px 25px rgba(255, 68, 68, 0.4);
        transform: translateY(-5px);
    }
    
    .food-card-title {
        color: #ffffff;
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 12px;
    }
    
    .food-card-description {
        color: #b0b0b0;
        font-size: 14px;
        line-height: 1.6;
        margin-bottom: 15px;
        min-height: 60px;
    }
    
    .food-card-rating {
        display: inline-block;
        background: linear-gradient(135deg, #ff4444 0%, #ffaa00 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 15px;
        margin-right: 10px;
    }
    
    .food-card-price {
        display: inline-block;
        background: #2a2a2a;
        color: #ffaa00;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 15px;
    }
    
    .food-card-category {
        display: inline-block;
        background: #2a2a2a;
        color: #ff4444;
        padding: 6px 12px;
        border-radius: 15px;
        font-weight: 600;
        font-size: 12px;
        margin-bottom: 10px;
    }
    
    .food-card-meta {
        color: #666;
        font-size: 12px;
        margin-top: 10px;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #ff4444 0%, #ffaa00 100%);
        color: white;
        font-weight: 700;
        border: none;
        padding: 15px 40px;
        border-radius: 10px;
        font-size: 18px;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        box-shadow: 0 8px 25px rgba(255, 68, 68, 0.5);
        transform: translateY(-2px);
    }
    
    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #ff4444, #ffaa00, transparent);
        margin: 50px 0;
    }
    
    /* Select box */
    .stSelectbox label {
        color: #ffffff !important;
        font-weight: 600;
        font-size: 16px;
    }
    
    /* Info box */
    .info-box {
        background: #1a1a1a;
        border-left: 4px solid #ff4444;
        padding: 20px;
        border-radius: 8px;
        color: #cccccc;
        margin: 20px 0;
    }
    
    /* Recommendation section */
    .rec-section {
        background: #0f0f0f;
        padding: 40px;
        border-radius: 15px;
        margin: 30px 0;
        border: 2px solid #2a2a2a;
    }
</style>
""", unsafe_allow_html=True)

# ================================
# LOAD MODELS
# ================================
@st.cache_resource
def load_models():
    try:
        with open('models/popular.pkl', 'rb') as f:
            popular_df = pickle.load(f)
        with open('models/pt.pkl', 'rb') as f:
            pt = pickle.load(f)
        with open('models/menu.pkl', 'rb') as f:
            menu = pickle.load(f)
        with open('models/similarity_scores.pkl', 'rb') as f:
            similarity_scores = pickle.load(f)
        
        return popular_df, pt, menu, similarity_scores
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

# ================================
# RECOMMENDATION FUNCTION
# ================================
def recommend(item_name, pt, menu, similarity_scores):
    try:
        index = np.where(pt.index == item_name)[0][0]
        similar_items = sorted(
            list(enumerate(similarity_scores[index])), 
            key=lambda x: x[1], 
            reverse=True
        )[1:6]
        
        recommendations = []
        for idx, score in similar_items:
            item = pt.index[idx]
            temp_df = menu[menu['ItemName'] == item].drop_duplicates('ItemName')
            
            if len(temp_df) > 0:
                recommendations.append({
                    'ItemName': temp_df['ItemName'].values[0],
                    'Description': temp_df['Description'].values[0] if 'Description' in temp_df.columns else 'N/A',
                    'Price': temp_df['Price'].values[0] if 'Price' in temp_df.columns else 0,
                    'Category': temp_df['Category'].values[0] if 'Category' in temp_df.columns else 'N/A',
                    'SimilarityScore': score
                })
        
        return recommendations
    except Exception as e:
        return []

# ================================
# DISPLAY FOOD CARD
# ================================
def display_food_card(item_name, description, rating=None, price=0, category="", num_ratings=None, similarity_score=None):
    """Display a food card with all details"""
    
    # Category badge
    category_badge = f'<div class="food-card-category">📂 {category}</div>' if category else ''
    
    # Rating/Similarity badge
    if similarity_score is not None:
        badge = f'<span class="food-card-rating">🎯 {similarity_score:.1%} Match</span>'
    elif rating is not None:
        badge = f'<span class="food-card-rating">⭐ {rating:.2f}/7</span>'
    else:
        badge = ''
    
    # Meta info
    meta = f'<div class="food-card-meta">👥 {num_ratings} ratings</div>' if num_ratings else ''
    
    card_html = f"""
    <div class="food-card">
        {category_badge}
        <div class="food-card-title">🍽️ {item_name}</div>
        <div class="food-card-description">{description}</div>
        <div style="margin-top: 15px;">
            {badge}
            <span class="food-card-price">💰 ₹{price:.2f}</span>
        </div>
        {meta}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

# ================================
# MAIN APP
# ================================
def main():
    # Load data
    popular_df, pt, menu, similarity_scores = load_models()
    
    if all([popular_df is not None, pt is not None, menu is not None, similarity_scores is not None]):
        
        # =============================
        # HEADER SECTION
        # =============================
        st.markdown('<div class="main-title">🍽️ Food Recommendation System</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Discover your next favorite dish powered by AI</div>', unsafe_allow_html=True)
        
        # =============================
        # SECTION 1: POPULAR ITEMS
        # =============================
        st.markdown('<div class="section-header">🌟 Popular Items</div>', unsafe_allow_html=True)
        
        # Get top popular items
        top_popular = popular_df.sort_values('avg_ratings', ascending=False).head(12)
        
        # Display in 3 columns
        cols = st.columns(3)
        for idx, (_, row) in enumerate(top_popular.iterrows()):
            with cols[idx % 3]:
                display_food_card(
                    row['ItemName'],
                    row['Description'],
                    rating=row['avg_ratings'],
                    price=row['Price'],
                    category=row.get('Category', 'N/A'),
                    num_ratings=row['num_ratings']
                )
        
        # Divider
        st.markdown("---")
        
        # =============================
        # SECTION 2: GET RECOMMENDATIONS
        # =============================
        st.markdown('<div class="section-header">🔍 Get Personalized Recommendations</div>', unsafe_allow_html=True)
        
        # Recommendation section
        # st.markdown('<div class="rec-section">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            all_items = sorted(pt.index.tolist())
            selected_item = st.selectbox(
                "Select a dish you like and we'll find similar ones:",
                all_items,
                key="recommender_select"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            recommend_btn = st.button("🎯 Find Similar", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show recommendations
        if recommend_btn or 'show_recommendations' in st.session_state:
            st.session_state['show_recommendations'] = True
            
            with st.spinner("Finding similar dishes..."):
                recommendations = recommend(selected_item, pt, menu, similarity_scores)
                
                if recommendations:
                    st.success(f"✨ Found {len(recommendations)} similar items to **{selected_item}**")
                    
                    # Display recommendations in 3 columns
                    cols = st.columns(3)
                    for idx, rec in enumerate(recommendations):
                        with cols[idx % 3]:
                            display_food_card(
                                rec['ItemName'],
                                rec['Description'],
                                price=rec['Price'],
                                category=rec['Category'],
                                similarity_score=rec['SimilarityScore']
                            )
                else:
                    st.error("No recommendations found. Try another item!")
        
        # Divider
        st.markdown("---")
        
        # =============================
        # SECTION 3: CATEGORY-WISE ITEMS
        # =============================
        st.markdown('<div class="section-header">📊 Browse by Category</div>', unsafe_allow_html=True)
        
        # Define category order and emojis
        category_info = {
            'Appetizers': '🥗',
            'Breakfast': '🍳',
            'Lunch': '🍱',
            'Dinner': '🍽️',
            'Desserts': '🍰',
            'Beverages': '🥤'
        }
        
        # Check if Category column exists
        if 'Category' in popular_df.columns:
            # Display each category
            for category, emoji in category_info.items():
                # Filter items for this category
                category_items = popular_df[popular_df['Category'] == category].sort_values('avg_ratings', ascending=False).head(6)
                
                if len(category_items) > 0:
                    st.markdown(f'<div class="category-header">{emoji} {category} ({len(category_items)} items)</div>', unsafe_allow_html=True)
                    
                    # Display in 3 columns
                    cols = st.columns(3)
                    for idx, (_, row) in enumerate(category_items.iterrows()):
                        with cols[idx % 3]:
                            display_food_card(
                                row['ItemName'],
                                row['Description'],
                                rating=row['avg_ratings'],
                                price=row['Price'],
                                category=row['Category'],
                                num_ratings=row['num_ratings']
                            )
        else:
            st.error("❌ Category column not found in popular_df!")
            st.markdown("""
            <div class="info-box">
                <strong>To fix this:</strong><br>
                1. Update your notebook to include 'Category' in popular_df<br>
                2. Change this line:<br>
                <code>popular_df = popular_df.merge(menu, on='ItemName').drop_duplicates('ItemName')[['ItemName', 'Category', 'Description', 'Price', 'ImageURL', 'num_ratings', 'avg_ratings']]</code><br>
                3. Re-run the notebook to regenerate pickle files
            </div>
            """, unsafe_allow_html=True)
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666; padding: 40px 0;'>
            <p style='font-size: 16px;'>🍽️ Powered by Collaborative Filtering | Made with ❤️ using Streamlit</p>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        st.error("❌ Failed to load models. Please ensure all pickle files exist in the 'models/' directory.")
        st.info("""
        **Required files:**
        - models/popular.pkl
        - models/pt.pkl
        - models/menu.pkl
        - models/similarity_scores.pkl
        """)

if __name__ == "__main__":
    main()
