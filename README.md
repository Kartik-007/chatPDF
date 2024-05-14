# chatPDF Application

## Introduction

The chatPDF application is an innovative Python-based tool designed to interactively communicate with the content of PDF documents. By leveraging advanced natural language processing (NLP) techniques, chatPDF enables users to ask questions directly to multiple PDF files, receiving accurate and contextually relevant answers. This application integrates powerful language models to comprehend and respond to queries by extracting and analyzing the text within your documents.

***

## How It Works

**chatPDF Workflow:**

1. **PDF Loading:** Users can upload one or more PDF documents into the application. chatPDF extracts the text from these documents for processing.
   
2. **Text Chunking:** To manage and analyze the content efficiently, the application divides the extracted text into smaller, manageable chunks.
   
3. **Embedding Generation:** chatPDF utilizes either OpenAI or Hugging Face models to create vector embeddings of the text chunks, enabling semantic analysis.
   
4. **Similarity Matching:** Upon receiving a query, the application identifies the text chunks that are most semantically related to the question posed.
   
5. **Response Generation:** The relevant text chunks are synthesized by the language model to craft a coherent and contextually relevant response based on the PDF content.


![screenshot](https://github.com/Kartik-007/chatPDF/blob/main/Blank%20diagram.png)
