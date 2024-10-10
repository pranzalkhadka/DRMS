# import streamlit as st
# import pytesseract
# from PIL import Image
# from Classification.predict import predict_category

# st.set_page_config(layout="wide")

# st.markdown(
#     """
#     <style>

#     #MainMenu {visibility: hidden;}
#     footer {visibility: hidden;}
#     header {visibility: hidden;}

#      /* Custom header sizes */
#     .small-header {
#         font-size: 1.8em; /* Decreased size */
#     }
#     .smaller-header {
#         font-size: 1.2em; /* Decreased size */
#     }

#     </style>
#     """,
#     unsafe_allow_html=True
# )

# def process_document(uploaded_file):
#     text = pytesseract.image_to_string(Image.open(uploaded_file), lang="nep-fuse-2") 
#     category = predict_category(text)
#     return text, category


# stored_documents = []

# st.markdown("<h1 class='small-header'>Document and Record Management System (DRMS)</h1>", unsafe_allow_html=True)

# st.markdown("<h2 class='smaller-header'>Upload Document</h2>", unsafe_allow_html=True)
# uploaded_file = st.file_uploader("Choose a document...", type=[ "jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     text, category = process_document(uploaded_file)
    
#     stored_documents.append({"filename": uploaded_file.name,"category": category})
#     st.success("Document uploaded and processed!")

# # Search section
# st.markdown("<h2 class='smaller-header'>Search Documents</h2>", unsafe_allow_html=True)
# search_query = st.text_input("Enter search text")

# # Initialize selected categories
# if 'selected_categories' not in st.session_state:
#     st.session_state.selected_categories = {"Policy": False, "Press Release": False, "Education": False, "ID": False}

# # Buttons for category selection arranged horizontally
# categories = ["Policy", "Press Release", "Education", "ID"]
# cols = st.columns(len(categories))

# for i, category in enumerate(categories):
#     with cols[i]:
#         if st.button(category, key=category):
#             st.session_state.selected_categories[category] = not st.session_state.selected_categories[category]

# # Determine which categories are selected
# selected_categories = [cat for cat, selected in st.session_state.selected_categories.items() if selected]

# # Filter documents based on search query and selected categories
# filtered_documents = []
# for doc in stored_documents:
#     if (not selected_categories or doc['category'] in selected_categories) and search_query.lower() in doc['text'].lower():
#         filtered_documents.append(doc)

# # Display stored documents
# st.markdown("<h2 class='smaller-header'>Stored Documents</h2>", unsafe_allow_html=True)
# if filtered_documents:
#     for doc in filtered_documents:
#         st.subheader(doc["filename"])
#         st.write(doc["text"])
#         st.write(f"Category: {doc['category']}")
# else:
#     st.write("No documents found.")
