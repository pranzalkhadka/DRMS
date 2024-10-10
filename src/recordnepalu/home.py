import streamlit as st
from streamlit_modal import Modal
import os
import subprocess, sys
import shutil
import pytesseract
from PIL import Image
from datetime import datetime
from Classification.predict import predict_category  # Import the predict_category function
from OCR.preprocess import preprocess_image
from OCR.highlighting import highlight_text
import time 
# Set the page configuration to wide layout
st.set_page_config(layout="wide")

# Custom CSS to hide the Streamlit menu bar, footer, and header, and remove the gap at the top
st.markdown(
    """
    <style>
    /* Hide the Streamlit menu bar, footer, and header */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Remove the gap at the top */
    .css-18e3th9 {
        padding-top: 0;
    }
    .css-1d391kg {
        padding-top: 0;
    }
    
    /* Custom header sizes */
    .small-header {
        font-size: 1.2em; /* Decreased size */
    }
    .smaller-header {
        font-size: 1em; /* Decreased size */
    }
      .stButton > button {
        margin: 2px 0;
        padding: 10px 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def process_document(uploaded_file):
    # Preprocess the image
    preprocessed_image = preprocess_image(uploaded_file)
    preprocessed_image_pil = Image.fromarray(preprocessed_image)
    text = pytesseract.image_to_string(preprocessed_image, lang="nep-fuse-2") 
    data = pytesseract.image_to_data(preprocessed_image, output_type=pytesseract.Output.STRING, lang='nep-fuse-2')
    pdf = pytesseract.image_to_pdf_or_hocr(preprocessed_image, extension='pdf', lang='nep-fuse-2')
    category = predict_category(text)
    return text, category, data, pdf, preprocessed_image_pil  # Return preprocessed image

def save_document(uploaded_file, text, category, data, pdf, preprocessed_image):
    # Create category directory if it doesn't exist
    category_dir = os.path.join('..', '..', 'results', 'documents', category)
    os.makedirs(category_dir, exist_ok=True)
    
    # Generate the timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    base_filename = f"original_{timestamp}"
    
    # Create a new folder for each document
    document_dir = os.path.join(category_dir, base_filename)
    os.makedirs(document_dir, exist_ok=True)
    
    # Save the original image
    image_filename = os.path.join(document_dir, f"{base_filename}{os.path.splitext(uploaded_file.name)[1]}")
    with open(image_filename, 'wb') as image_file:
        image_file.write(uploaded_file.getbuffer())
    
    # Save the preprocessed image
    preprocessed_image_filename = os.path.join(document_dir, f"{base_filename}_preprocessed.png")
    preprocessed_image.save(preprocessed_image_filename)  # Save preprocessed image using PIL

    # Save the text file
    text_filename = os.path.join(document_dir, f"{base_filename}.txt")
    with open(text_filename, 'w', encoding='utf-8') as text_file:
        text_file.write(text)
    
    # Save the PDF file
    pdf_filename = os.path.join(document_dir, f"{base_filename}.pdf")
    with open(pdf_filename, 'wb') as pdf_file:
        pdf_file.write(pdf)
    
    # Save the OCR data file
    ocr_data_filename = os.path.join(document_dir, f"{base_filename}_data.txt")
    with open(ocr_data_filename, 'w', encoding='utf-8') as ocr_data_file:
        ocr_data_file.write(data)
    
    return image_filename, text_filename, preprocessed_image_filename  # Return preprocessed image filename

def load_documents():
    documents = []
    # Navigate to 'results/documents'
    base_dir = os.path.join('..', '..', 'results', 'documents')
    
    # Check if the directory exists
    if not os.path.exists(base_dir):
        return documents
    
    for category in os.listdir(base_dir):
        category_dir = os.path.join(base_dir, category)
        if os.path.isdir(category_dir):
            for document_folder in os.listdir(category_dir):
                document_dir = os.path.join(category_dir, document_folder)
                if os.path.isdir(document_dir):
                    for filename in os.listdir(document_dir):
                        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                            image_path = os.path.join(document_dir, filename)
                            text_path = os.path.join(document_dir, f"{os.path.splitext(filename)[0]}.txt")
                            documents.append({
                                "filename": filename,
                                "image_path": image_path,
                                "text_path": text_path,
                                "category": category
                            })
    return documents

# Initialize session state for stored documents and document hashes
if 'stored_documents' not in st.session_state:
    st.session_state.stored_documents = load_documents()


# Create a modal instance
# modal = Modal(key="image_modal", title="Document Preview",padding=20)

# Use custom CSS classes for headers
st.markdown("<h1 class='small-header'> Nepali Document and Record Management System (DRMS)</h1>", unsafe_allow_html=True)

st.markdown("<h2 class='smaller-header'>Upload Document</h2>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose a document...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Calculate the hash of the uploaded file
    
    # Check if the hash already exists
        text, category, data, pdf, preprocessed_image = process_document(uploaded_file)
        image_path, text_path, preprocessed_image_filename = save_document(uploaded_file, text, category, data, pdf, preprocessed_image)
        
        # Store the hash and document details
        st.session_state.stored_documents.append({
            "filename": uploaded_file.name,
            "image_path": image_path,
            "text_path": text_path,
            "category": category
        })
        st.success("Document uploaded and processed!")
     
        

# Search section
st.markdown("<h2 class='smaller-header'>Search Documents</h2>", unsafe_allow_html=True)
search_query = st.text_input("Enter search text")

# Initialize selected categories without the "All" option
if 'selected_categories' not in st.session_state:
    st.session_state.selected_categories = {
        "Policy": False,
        "Press_Release": False,
        "Education": False,
        "ID": False
    }

# Display category checkboxes side by side
cols = st.columns(len(st.session_state.selected_categories))
for i, category in enumerate(st.session_state.selected_categories.keys()):
    st.session_state.selected_categories[category] = cols[i].checkbox(category, value=st.session_state.selected_categories[category])

# Filter documents based on search query and selected categories
selected_categories = [cat for cat, selected in st.session_state.selected_categories.items() if selected]

# Filter documents based on search query and selected categories
filtered_documents = []
for doc in st.session_state.stored_documents:
    if (not selected_categories or doc['category'] in selected_categories):
        # Check if text_path exists before trying to open it
        if os.path.exists(doc["text_path"]):
            if search_query:
                with open(doc["text_path"], 'r', encoding='utf-8') as file:
                    content = file.read()
                    if search_query.lower() in content.lower():
                        filtered_documents.append(doc)
            else:
                filtered_documents.append(doc)

@st.dialog("Document Preview")
def show_document_preview(doc):
    if search_query != '':
        base_name = os.path.splitext(doc["image_path"])[0]
        data_path = base_name + '_data.txt'
        preprocessed_path = base_name + '_preprocessed.png'
        output_img_path = base_name + '_temp.png'
        highlight_text(doc["image_path"], data_path, preprocessed_path, search_query, output_img_path)
        st.image(output_img_path, use_column_width=True)
        # Delete the temporary image file after showing it
        os.remove(output_img_path)
    else:
        st.image(doc["image_path"], use_column_width=True)
    
    buttons = st.columns(2)
    with buttons[0]:
        if os.path.exists(pdf_path):
            if st.button(label="Open PDF", key=f"pdf_{doc['filename']}_{i}"):
                # os.startfile(pdf_path)
                opener = "open" if sys.platform == "darwin" else "xdg-open"
                subprocess.call([opener, pdf_path])
    with buttons[1]:
        #code to delete the document folder
        if st.button(label="Delete Document", key=f"delete_{doc['filename']}_{i}"): 
            shutil.rmtree(os.path.dirname(doc["image_path"]))
            st.session_state.stored_documents.remove(doc)
            time.sleep(1)
            st.rerun()
    
    st.write(f"Category: {doc['category']}")


# Display stored documents
# st.markdown("<h2 class='smaller-header'>Stored Documents</h2>", unsafe_allow_html=True)
# if filtered_documents:
#     cols = st.columns(3)  # Adjust the number of columns as needed
#     for i, doc in enumerate(filtered_documents):
#         with cols[i % 3]:  # Cycle through columns
#             st.markdown(f"<h3 class='smaller-header'>{doc['filename']}</h3>", unsafe_allow_html=True)
#             st.image(doc['image_path'], use_column_width=False, width=150)
#             # Open the modal on click
#             if st.button("View Document", key=doc['filename']):
#                 show_document_preview(doc)
# else:
#     st.write("No documents found.")

st.markdown("<h2 class='smaller-header'>Stored Documents</h2>", unsafe_allow_html=True)

if filtered_documents:
    cols = st.columns(3)  # Adjust the number of columns as needed
    for i, doc in enumerate(filtered_documents):
        with cols[i % 3]:  # Cycle through columns
            st.markdown(f"<h3 class='smaller-header'>{doc['filename']}</h3>", unsafe_allow_html=True)
            st.image(doc['image_path'], use_column_width=False, width=150)
            st.write(f"Category: {doc['category']}")
            # Create pdf_path for the document
            pdf_path = os.path.splitext(doc["image_path"])[0] + '.pdf'  # Initialize pdf_path here
            
            # Create a row with two columns for the buttons
            
            if st.button("View Document", key=f"{doc['filename']}_view_{i}"):  # Ensure unique keys
                show_document_preview(doc)
            
else:
    st.write("No documents found.")
