import requests
import zipfile

def download_and_extract_data():
    file_id = "10n_2CmGWkckx29VanfMF8zjB1g99S5DQ"
    destination = "/home/pranjal/Downloads/Document-Management-of-Nepali-Papers/dataset/Classification/classification_data.zip"
    URL = "https://docs.google.com/uc?export=download&confirm=1"
    
    session = requests.Session()
    response = session.get(URL, params={"id": file_id})
    
    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break
    
    if token:
        params = {"id": file_id, "confirm": token}
        response = session.get(URL, params=params)
    
    with open(destination, "wb") as f:
        f.write(response.content)
    
    with zipfile.ZipFile(destination, 'r') as zip_ref:
        zip_ref.extractall("/home/pranjal/Downloads/Document-Management-of-Nepali-Papers/dataset/Classification")

if __name__ == "__main__":
    download_and_extract_data()