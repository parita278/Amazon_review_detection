import pandas as pd 
import pickle
import streamlit as st
import numpy as np
from scipy.sparse import hstack, csr_matrix

st.set_page_config(
    page_title="Amazon Review Prediction",
    layout='centered'
)

# Load the model dictionary
with open("Model/best_model.pkl","rb") as file:
    model_dict = pickle.load(file)

# Extract components from the dictionary
model = model_dict['model']
vectorizer = model_dict['vectorizer']
scaler = model_dict['scaler']
feature_names = model_dict['feature_names']

st.title("Amazon Review Prediction App")

if model is not None:
    st.markdown("<h2 style='text-align: center;'> Amazon Review</h2>", unsafe_allow_html=True)
    reviews = st.text_input("Enter the Reviews", key='reviews')
    rating = st.selectbox("Select Rating", [1, 2, 3, 4, 5], key='rating')
    
    if st.button(" Predict Review", type="primary"):
        try:
            # Create temporary dataframe for feature extraction
            temp_df = pd.DataFrame({
                'reviews.text': [reviews],
                'reviews.rating': [rating]
            })
            
            # Extract features (simplified version of the notebook's feature extraction)
            temp_df['review_length'] = temp_df['reviews.text'].str.len()
            temp_df['word_count'] = temp_df['reviews.text'].str.split().str.len()
            temp_df['exclamation_count'] = temp_df['reviews.text'].str.count('!')
            temp_df['capital_ratio'] = temp_df['reviews.text'].apply(lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1))
            temp_df['is_extreme_rating'] = ((temp_df['reviews.rating'] == 1) | (temp_df['reviews.rating'] == 5)).astype(int)
            
            # Sentiment analysis
            positive_words = ['good', 'great', 'excellent', 'amazing', 'perfect', 'love', 'best', 'awesome', 'recommend']
            negative_words = ['bad', 'terrible', 'awful', 'worst', 'hate', 'horrible', 'disappointing', 'poor', 'waste']
            
            temp_df['positive_word_count'] = temp_df['reviews.text'].str.lower().apply(lambda x: sum(1 for word in positive_words if word in str(x).split()))
            temp_df['negative_word_count'] = temp_df['reviews.text'].str.lower().apply(lambda x: sum(1 for word in negative_words if word in str(x).split()))
            
            # Rating-text mismatch detection
            temp_df['rating_text_mismatch'] = (
                ((temp_df['reviews.rating'] >= 4) & (temp_df['negative_word_count'] > temp_df['positive_word_count'])) |
                ((temp_df['reviews.rating'] <= 2) & (temp_df['positive_word_count'] > temp_df['negative_word_count']))
            ).astype(int)
            
            # Add default values for missing features
            temp_df['user_review_count'] = 1
            temp_df['user_avg_rating'] = temp_df['reviews.rating']
            temp_df['is_verified_purchase'] = 1
            temp_df['helpful_votes'] = 0
            
            # Transform text using the fitted vectorizer
            x_text = vectorizer.transform([reviews])
            
            # Get numerical features
            available_features = [col for col in feature_names if col in temp_df.columns]
            if available_features:
                x_numerical = temp_df[available_features].fillna(0).values
                x_numerical_scaled = scaler.transform(x_numerical)
            else:
                # Fallback features
                x_numerical = np.column_stack([
                    len(reviews),
                    len(reviews.split()),
                    rating
                ])
                x_numerical_scaled = scaler.transform(x_numerical)
            
            # Combine features
            x_combined = hstack([x_text, csr_matrix(x_numerical_scaled)])
            
            # Make prediction
            prediction = model.predict(x_combined)[0]
            probability = model.predict_proba(x_combined)[0]
            
            # Display results
            st.markdown("<h2> Prediction Result</h2>", unsafe_allow_html=True)
            
            if prediction == 1:
                st.markdown("<h2 style='color: red;'> FAKE REVIEW DETECTED</h2>", unsafe_allow_html=True)
            else:
                st.markdown("<h2 style='color: green;'> GENUINE REVIEW</h2>", unsafe_allow_html=True)
            
            # Show confidence
            confidence = max(probability) * 100
            fake_probability = probability[1] * 100 if len(probability) > 1 else probability[0] * 100
            
            st.markdown(f"**Confidence:** {confidence:.1f}%")
            st.markdown(f"**Fake Probability:** {fake_probability:.1f}%")

            # sentiment value based on the ratings from the user
            if rating == 3:
                st.markdown("<h3 style='color: orange'>Neutral Review</h3>",unsafe_allow_html=True)
            elif rating == 1 or rating == 2:
                st.markdown("<h3 style='color: red'>Negative Review</h3>",unsafe_allow_html=True)
            elif rating == 4 or rating == 5:
                st.markdown("<h3 style='color: green'>Positive Review</h3>",unsafe_allow_html=True)


            # # Show review analysis
            # st.markdown("###  Review Analysis")
            # col1, col2 = st.columns(2)
            
            # with col1:
            #     st.metric("Review Length", f"{len(reviews)} characters")
            #     st.metric("Word Count", f"{len(reviews.split())} words")
            #     st.metric("Exclamation Count", f"{reviews.count('!')}")
            
            # with col2:
            #     st.metric("Capital Ratio", f"{temp_df['capital_ratio'].iloc[0]:.2f}")
            #     st.metric("Positive Words", f"{temp_df['positive_word_count'].iloc[0]}")
            #     st.metric("Negative Words", f"{temp_df['negative_word_count'].iloc[0]}")

        except Exception as e:
            st.error(f" Error making prediction: {str(e)}")
            st.error("Please check your input and try again.")

else:
    st.error(" Model not loaded properly")
