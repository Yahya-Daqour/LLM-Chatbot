import re
from llama_index.core import Document

class TextCleaner:
    def __init__(self):
        pass

    def clean_text_content(self, raw_text: str) -> list:
        """
        Cleans and structures text from starting pages or general pages.
        Handles metadata, lists, and narrative content dynamically.
        
        Args:
            raw_text (str): Raw text input from a page.
        Returns:
            list: List of structured segments with metadata, lists, and narrative.
        """
        # Step 1: Remove excessive spaces and line breaks
        cleaned_text = re.sub(r'\s+', ' ', raw_text).strip()

        # Initialize list for structured content
        segments = []

        # Step 2: Detect and extract ISBN
        isbn_match = re.search(r'ISBN[\s:-]*([\d-]+)', cleaned_text, re.IGNORECASE)
        if isbn_match:
            segments.append({"type": "metadata", "content": f"ISBN: {isbn_match.group(1)}"})

        # Step 3: Detect Author or Contributor Name
        author_match = re.search(r'Center[\s\n]*(.*?)\n', cleaned_text, re.IGNORECASE)
        if author_match:
            segments.append({"type": "metadata", "content": f"Author: {author_match.group(1).strip()}"})

        # Step 4: Remove Arabic characters
        cleaned_text = re.sub(r'[\u0600-\u06FF]+', '', cleaned_text)

        # Step 5: Remove irrelevant text patterns
        cleaned_text = re.sub(r"WC\d?[_\w\d]+\.indb", "", cleaned_text)
        cleaned_text = re.sub(r"\b\d{1,2}\b", "", cleaned_text)  # Matches standalone numbers like 1, 16, etc.
        cleaned_text = re.sub(r"(PM|AM|[0-9]{1,2}:[0-9]{2})", "", cleaned_text)  # Matches timestamps
        cleaned_text = re.sub(r"www\.[^\s]+|http[^\s]+", "", cleaned_text)  # Matches URLs
        cleaned_text = re.sub(r'[^\w\s.,:!/\'-]', '', cleaned_text)  # Remove remaining special characters except :,/, and .

        # Step 6: Remove standalone symbols (e.g., `,`, `.` as individual tokens)
        cleaned_text = re.sub(r'^[\.,!?\'-]', '', cleaned_text)  # Removes standalone punctuation marks

        # Step 7: Remove duplicate words
        cleaned_text = re.sub(r'\b(\w+)\s+\1\b', r'\1', cleaned_text)

        # Step 8: Extract bullet points (e.g., lists with symbols)
        bullets = re.findall(r'[\x81•\-]\s*(.*?)\s*(?=[\x81•\-]|$)', cleaned_text)
        for bullet in bullets:
            segments.append({"type": "list_item", "content": bullet.strip()})

        # Step 9: Remove bullets and metadata to isolate narrative content
        cleaned_text_no_bullets = re.sub(r'[\x81•\-].*?\s*(?=[\x81•\-]|$)', '', cleaned_text).strip()
        narrative = re.sub(r'(ISBN[\s:-]*[\d-]+|Center[\s\n]*.*?)', '', cleaned_text_no_bullets, flags=re.IGNORECASE).strip()

        # Step 10: Split long text into sentences for readability
        if narrative:
            sentences = re.split(r'(?<=[.!?])\s+', narrative)
            for sentence in sentences:
                if sentence.strip():
                    segments.append({"type": "narrative", "content": sentence.strip()})

        # Step 11: Remove lines that contain only a punctuation mark
        cleaned_text_lines = cleaned_text.split('\n')
        cleaned_text_lines = [line for line in cleaned_text_lines if not re.match(r'^\s*[.,!?\'-]\s*$', line)]
        cleaned_text = '\n'.join(cleaned_text_lines)

        # Update the narrative content
        narrative_lines = narrative.split('\n')
        narrative_lines = [line.lstrip() for line in narrative_lines if not re.match(r'^\s*[.,!?\'-]\s*$', line)]
        narrative = '\n'.join(narrative_lines)

        return segments


class DocumentProcessor:
    def __init__(self, text_cleaner: TextCleaner):
        self.text_cleaner = text_cleaner

    def process_documents(self, documents: list) -> list:
        """
        Cleans and processes a list of Document objects.
        
        Args:
            documents (list): A list of Document objects.
        
        Returns:
            list: A list of cleaned Document objects ready for indexing.
        """
        cleaned_documents = []
        
        for doc in documents:
            # Extract and clean the text content from the Document
            raw_text = doc.text
            cleaned_data = self.text_cleaner.clean_text_content(raw_text)
            
            # Combine cleaned segments into a single text string
            combined_content = "\n".join(
                segment['content']
                for segment in cleaned_data
                if segment['type'] in ['metadata', 'list_item', 'narrative']
            )
            
            # Create a new Document object with the cleaned content
            cleaned_documents.append(Document(text=combined_content))
        
        return cleaned_documents