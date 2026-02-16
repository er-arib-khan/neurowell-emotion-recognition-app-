import streamlit as st
from datetime import datetime

def test_text_analysis_save():
    """Test function to verify text analysis saving"""
    
    # Create test text analysis data
    test_data = {
        'type': 'text_analysis',
        'sentiment': 'positive',
        'dominant_emotion': 'positive',
        'text': 'This is a test analysis entry',
        'word_count': 5,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'scores': {'compound': 0.5, 'pos': 0.6, 'neu': 0.3, 'neg': 0.1}
    }
    
    # Initialize session state if needed
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    
    # Add test data
    st.session_state.analysis_history.append(test_data)
    
    st.success("Test data added successfully!")
    st.write(f"Current history has {len(st.session_state.analysis_history)} entries")
    
    # Show the added data
    st.json(test_data)

if __name__ == "__main__":
    st.set_page_config(page_title="Test Text Analysis")
    st.title("🧪 Text Analysis Test")
    
    if st.button("Add Test Data"):
        test_text_analysis_save()
    
    if st.button("Show Current History"):
        if 'analysis_history' in st.session_state:
            st.write(f"Total entries: {len(st.session_state.analysis_history)}")
            for i, entry in enumerate(st.session_state.analysis_history):
                st.write(f"Entry {i}: {entry.get('type', 'unknown')}")
        else:
            st.write("No history in session state")