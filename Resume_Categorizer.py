import streamlit as st
import re
import pickle
import fitz
from streamlit_pdf_viewer import pdf_viewer

st.title('Resume Categorizer')
st.write('')
st.text('Upload resume to check which category it belongs to !')
st.write('')
st.write('')

def cleanResumeText(text):
    cleanText = re.sub('\s+', ' ', text)
    cleanText = re.sub(r'http\S+|[\*/\(\)\-\.\,]|[^\x00-\x7f]', '', cleanText).lower()
    return cleanText

tfidf = pickle.load(open('tfidf.pkl', 'rb'))
svm_clf = pickle.load(open('svm_clf.pkl', 'rb'))

category_map = {
    6: 'Data Science',
    12: 'HR',
    0: 'Advocate',
    1: 'Arts',
    24: 'Web Designing',
    16: 'Mechanical Engineer',
    22: 'Sales',
    14: 'Health and Fitness',
    5: 'Civil Enigneer',
    15: 'Web Developer',
    4: 'Business Analyst',
    21: 'SAP Developer',
    2: 'Automation Testing',
    11: 'Electrical Engineer',
    18: 'Operations Manager',
    20: 'Python Developer',
    8: 'DevOps Engineer',
    17: 'Network Security Engineer',
    19: 'PMO',
    7: 'Database Engineer',
    13: 'Big Data Engineer',
    10: 'ETL Developer',
    9: 'DotNet Developer',
    3: 'Blockchain Developer',
    23: 'Tester'
}

def read_pdf(file):
    pdf_document = fitz.open(stream=file.read(), filetype='pdf')
    text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

uploaded_file = st.file_uploader("Choose a file", type='pdf')
if uploaded_file is not None:
    try:
        binary_data = uploaded_file.getvalue()
        resumeText = read_pdf(uploaded_file)
        resumeText = cleanResumeText(resumeText)
        vectorizeText = tfidf.transform([resumeText])
        predict = svm_clf.predict(vectorizeText)
        st.write('')
        st.subheader(f'The resume belongs to the category of: {category_map[predict[0]]}')
        st.write('')
        pdf_viewer(input=binary_data, width=700)
    except Exception as e:
        st.write(e)