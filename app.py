!pip install langchain
!pip install PyPDF2
!pip install transformers

from transformers import AutoTokenizer, pipeline
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import HuggingFaceDatasetLoader
from langchain.llms.huggingface_pipeline import HuggingFacePipeline


def main():

    st.title("Question Answering Bot")
    st.subheader("Using SQuAD dataset and distilbert model")

    pdf = st.file_uploader("**Or upload your PDF:**", type='pdf')
    use_text = 1

    if pdf is not None:
        use_text = 0
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)
        embeddings = HuggingFaceEmbeddings(model_name='bert-base-cased')
        db = FAISS.from_texts(chunks, embeddings)
        st.write('Ready to answer your question')

    text = st.text_area("**Or enter your text:**", "")
    question = st.text_input("Enter your question:", "")

    if st.button("Submit"):
        model_checkpoint = "katxtong/distilbert-base-uncased-finetuned-squad"
        question_answerer = pipeline(
            "question-answering", model=model_checkpoint)

        if use_text == 1:
            context = text

        else:
            searchDocs = db.similarity_search(question)
            text = searchDocs[0].page_content

        ans = question_answerer(question=question, context=text)['answer']
        st.subheader("Answer: ")
        st.write(ans)


if __name__ == "__main__":
    main()


# sample text:
# New York (CNN) -- More than 80 Michael Jackson collectibles -- including the late pop star's famous rhinestone-studded glove from a 1983 performance -were auctioned off Saturday, reaping a total $2 million. Profits from the auction at the Hard Rock Cafe in New York's Times Square crushed pre-sale expectations of only $120,000 in sales. The highly prized memorabilia, which included items spanning the many stages of Jackson's career, came from more than 30 fans, associates and family members, who contacted Julien's Auctions to sell their gifts and mementos of the singer.  Jackson's flashy glove was the big-ticket item of the night, fetching $420,000 from a buyer in Hong Kong, China. Jackson wore the glove at a 1983 performance during \"Motown 25,\" an NBC special where  he debuted his revolutionary moonwalk. Fellow Motown star Walter \"Clyde\" Orange of the Commodores,  who also performed in the special 26 years ago, said he asked for Jackson's autograph at the time,  but Jackson gave him the glove instead. "The legacy that [Jackson] left behind is bigger than life for me,\"Orange said. \"I hope that through that glove people can see what he was trying to say in his music and what he said in his music.\" Orange said he plans to give a portion of the proceeds to charity. Hoffman Ma, who bought the glove on behalf of Ponte 16 Resort in Macau, paid a 25 percent buyer's premium, which was tacked onto all final sales over $50,000. Winners of items less than $50,000 paid a 20 percent premium.
