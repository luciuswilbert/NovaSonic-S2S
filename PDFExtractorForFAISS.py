import fitz  # PyMuPDF
import os
import json
import re
from typing import List, Dict, Any
from pathlib import Path

class PDFExtractorForFAISS:
    def __init__(self, pdf_folder: str = "DocumentsPDF", output_folder: str = "extractedPDFs"):
        """
        Initialize the PDF extractor for FAISS preparation.
        
        Args:
            pdf_folder (str): Folder containing PDF files
            output_folder (str): Folder to save extracted text and metadata
        """
        self.pdf_folder = pdf_folder
        self.output_folder = output_folder
        self.chunk_size = 1000  # Characters per chunk
        self.chunk_overlap = 200  # Overlap between chunks
        
        # Create output folder if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)
        
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.
        
        Args:
            text (str): Raw extracted text
            
        Returns:
            str: Cleaned text
        """
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove special characters that might cause issues
        text = re.sub(r'[^\w\s\-\.\,\!\?\:\;\(\)\[\]\{\}\/\+\=\*\&\^\%\$\#\@]', '', text)
        
        return text
    
    def chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """
        Split text into overlapping chunks for better embedding.
        
        Args:
            text (str): Text to chunk
            chunk_size (int): Size of each chunk
            overlap (int): Overlap between chunks
            
        Returns:
            List[str]: List of text chunks
        """
        if chunk_size is None:
            chunk_size = self.chunk_size
        if overlap is None:
            overlap = self.chunk_overlap
            
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # If this is not the last chunk, try to break at a sentence boundary
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                search_start = max(start, end - 100)
                sentence_end = text.rfind('.', search_start, end)
                if sentence_end > start + chunk_size // 2:  # Only break if we find a good boundary
                    end = sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
            
            # Move start position for next chunk
            start = end - overlap
            if start >= len(text):
                break
                
        return chunks
    
    def extract_pdf_text(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract text from a single PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            Dict[str, Any]: Dictionary containing extracted text and metadata
        """
        try:
            doc = fitz.open(pdf_path)
            
            # Extract basic metadata
            metadata = {
                "filename": os.path.basename(pdf_path),
                "filepath": pdf_path,
                "num_pages": len(doc),
                "file_size": os.path.getsize(pdf_path)
            }
            
            # Extract text from all pages
            full_text = ""
            page_texts = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text("text")
                cleaned_text = self.clean_text(text)
                
                if cleaned_text:
                    full_text += f"\n--- Page {page_num + 1} ---\n"
                    full_text += cleaned_text
                    full_text += "\n"
                    
                    page_texts.append({
                        "page_num": page_num + 1,
                        "text": cleaned_text,
                        "char_count": len(cleaned_text)
                    })
            
            doc.close()
            
            # Clean the full text
            full_text = self.clean_text(full_text)
            
            # Create chunks for FAISS
            chunks = self.chunk_text(full_text)
            
            return {
                "metadata": metadata,
                "full_text": full_text,
                "page_texts": page_texts,
                "chunks": chunks,
                "total_chunks": len(chunks),
                "total_characters": len(full_text)
            }
            
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return None
    
    def process_all_pdfs(self) -> Dict[str, Any]:
        """
        Process all PDF files in the specified folder.
        
        Returns:
            Dict[str, Any]: Summary of processing results
        """
        pdf_files = []
        for file in os.listdir(self.pdf_folder):
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(self.pdf_folder, file))
        
        print(f"Found {len(pdf_files)} PDF files to process")
        print("=" * 60)
        
        all_extractions = {}
        total_chunks = 0
        total_characters = 0
        
        for pdf_path in pdf_files:
            print(f"Processing: {os.path.basename(pdf_path)}")
            
            extraction = self.extract_pdf_text(pdf_path)
            if extraction:
                filename = os.path.basename(pdf_path)
                all_extractions[filename] = extraction
                
                total_chunks += extraction['total_chunks']
                total_characters += extraction['total_characters']
                
                print(f"  ✓ Extracted {extraction['total_chunks']} chunks")
                print(f"  ✓ Total characters: {extraction['total_characters']:,}")
                
                # Save individual file extraction
                output_file = os.path.join(self.output_folder, f"{filename}_extraction.json")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(extraction, f, indent=2, ensure_ascii=False)
                
                # Save individual file text
                text_file = os.path.join(self.output_folder, f"{filename}_text.txt")
                with open(text_file, 'w', encoding='utf-8') as f:
                    f.write(extraction['full_text'])
            else:
                print(f"  ✗ Failed to extract text")
        
        # Create summary and save
        summary = {
            "total_files": len(pdf_files),
            "successful_extractions": len(all_extractions),
            "total_chunks": total_chunks,
            "total_characters": total_characters,
            "files_processed": list(all_extractions.keys())
        }
        
        # Save summary
        summary_file = os.path.join(self.output_folder, "extraction_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Save all extractions in one file
        all_extractions_file = os.path.join(self.output_folder, "all_extractions.json")
        with open(all_extractions_file, 'w', encoding='utf-8') as f:
            json.dump(all_extractions, f, indent=2, ensure_ascii=False)
        
        # Create FAISS-ready format
        faiss_data = self.prepare_faiss_data(all_extractions)
        faiss_file = os.path.join(self.output_folder, "faiss_ready_data.json")
        with open(faiss_file, 'w', encoding='utf-8') as f:
            json.dump(faiss_data, f, indent=2, ensure_ascii=False)
        
        print("\n" + "=" * 60)
        print("EXTRACTION SUMMARY:")
        print("=" * 60)
        print(f"Total PDF files: {summary['total_files']}")
        print(f"Successfully processed: {summary['successful_extractions']}")
        print(f"Total text chunks: {summary['total_chunks']:,}")
        print(f"Total characters: {summary['total_characters']:,}")
        print(f"Output folder: {self.output_folder}")
        
        return summary
    
    def prepare_faiss_data(self, extractions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Prepare data in a format ready for FAISS embedding.
        
        Args:
            extractions (Dict[str, Any]): All PDF extractions
            
        Returns:
            List[Dict[str, Any]]: FAISS-ready data with chunks and metadata
        """
        faiss_data = []
        
        for filename, extraction in extractions.items():
            for i, chunk in enumerate(extraction['chunks']):
                faiss_item = {
                    "id": f"{filename}_chunk_{i}",
                    "text": chunk,
                    "metadata": {
                        "filename": filename,
                        "chunk_index": i,
                        "total_chunks": extraction['total_chunks'],
                        "file_metadata": extraction['metadata']
                    }
                }
                faiss_data.append(faiss_item)
        
        return faiss_data
    
    def print_sample_chunks(self, num_chunks: int = 3):
        """
        Print sample chunks from the extracted data.
        
        Args:
            num_chunks (int): Number of sample chunks to print
        """
        faiss_file = os.path.join(self.output_folder, "faiss_ready_data.json")
        
        if not os.path.exists(faiss_file):
            print("No FAISS data found. Please run process_all_pdfs() first.")
            return
        
        with open(faiss_file, 'r', encoding='utf-8') as f:
            faiss_data = json.load(f)
        
        print(f"\nSAMPLE CHUNKS (showing {min(num_chunks, len(faiss_data))} of {len(faiss_data)}):")
        print("=" * 80)
        
        for i, item in enumerate(faiss_data[:num_chunks]):
            print(f"\nChunk {i+1}:")
            print(f"ID: {item['id']}")
            print(f"Filename: {item['metadata']['filename']}")
            print(f"Text preview: {item['text'][:200]}...")
            print("-" * 40)

def main():
    """Main function to run the PDF extraction process."""
    print("PDF Extractor for FAISS Embedding")
    print("=" * 50)
    
    # Initialize extractor
    extractor = PDFExtractorForFAISS()
    
    # Process all PDFs
    summary = extractor.process_all_pdfs()
    
    # Show sample chunks
    extractor.print_sample_chunks()
    
    print(f"\nAll files saved to: {extractor.output_folder}")
    print("Files created:")
    print("  - extraction_summary.json (overall summary)")
    print("  - all_extractions.json (complete data)")
    print("  - faiss_ready_data.json (FAISS-ready format)")
    print("  - {filename}_extraction.json (individual file data)")
    print("  - {filename}_text.txt (individual file text)")

if __name__ == "__main__":
    main() 