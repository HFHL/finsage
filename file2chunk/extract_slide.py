import json
import os
import time
from openai import OpenAI
import base64
from typing import List, Dict, Tuple
import logging
from tqdm import tqdm
import pymupdf

# Set up logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('slide_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def encode_image_to_base64(image_path: str) -> str:
    """Encode image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def slide_sys_prompt():
    return f'''
You are a specialized assistant for converting slide graph content into well-structured text and generating contextual summaries. Here's how you should process the content:

Inputs:
[Graph]
The graph from the current slide page
[Context]
Integrated summary from previous slide pages (if available)

Processing Instructions
Part 1: Content Parsing

Convert bullet points and hierarchical structures into cohesive paragraphs
Maintain logical flow and connections between ideas
Expand abbreviations and technical terms when necessary
Preserve all the numerical data and statistics
Convert any graphical representations into descriptive text

Part 2: Contextual Summary Generation
Generate an integrated summary that:

Integrate and briefly state the topic from the previous slide's context summary and the current slide's content, but avoid repeating numerical data

Output format:

[CHUNK]
{{paragraph 1 content}}

[CHUNK]
{{paragraph 2 content}}
...

[Summary]
{{contextual summary}}

If no paragraph can be parsed, provide only the context summary in the format:

[Summary]
{{contextual summary}}

Example Output:
[CHUNK]
In the first nine months of 2024, Lotus reported significant achievements. Over 7,600 vehicles were delivered, representing a 136% year-on-year increase compared to 3,221 vehicles delivered during the same period in 2023. The third quarter alone saw 2,744 vehicles delivered, a 54% increase compared to the 1,782 vehicles delivered in Q3 2023.

[CHUNK]
Total revenue for the first nine months of 2024 reached $653 million, more than double the $318 million revenue in the first nine months of 2023, marking a 105% year-on-year growth. Revenue for Q3 2024 alone stood at $255 million, a 36% increase compared to $188 million in Q3 2023.

[CHUNK]
The intelligent driving business also showed remarkable progress, with revenue from customers outside Lotus soaring to $11 million, a year-on-year growth of 450%. Gross profit margins, however, declined. For Q3 2024, the margin was 3%, a significant drop from 15% in Q3 2023. The nine-month gross profit margin for 2024 was 9%, lower than the 11% recorded for the same period in 2023.

[Summary]
The topic is Q3 2024 and the first nine months 2024 results of Lotus Technology. The disclaimer states that the document is for information only and not a basis for investment decisions. Lotus achieved robust growth in vehicle deliveries and revenue in the first nine months of 2024. The intelligent driving business exhibited exceptional growth from external customers. However, gross profit margins declined, reflecting cost challenges despite the sales surge.
'''

def slide_user_prompt(context: str):
    ret = f'Please process this slide according to the given instructions.\n[Context]\n{context}'
    return ret

def extract_slide_content(client: OpenAI, image_path: str, ctx: str) -> str:
    """Extract slide content using GPT-4 Vision."""
    base64_image = encode_image_to_base64(image_path)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": slide_sys_prompt()
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": slide_user_prompt(ctx)
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            max_tokens=3000
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error in GPT API call: {e}")
        return ""

def create_chunks(content: str, page_num: int) -> List[Dict]:
    """Create chunks from slide content with context."""
    chunks = []
    if "[CHUNK]" not in content:
        return []
    
    contents = content.split("[CHUNK]")

    for c in contents:
        c = c.strip()
        if c == "":
            continue
        chunk_dict = {
            'content': c,
            'page_number': page_num,
            "type": "slide"
        }
        chunks.append(chunk_dict)
        print(c)
    return chunks
        
def process_slides(input_path: str, output_json_path: str, start: int, end: int):
    """Main pipeline to process slides and create chunks."""
    # Load the PDF file
    pdf_file = pymupdf.open(input_path)
    all_chunks = []
    ctx = ""
    client = OpenAI(
        api_key="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",
    )

    for page_idx in range(len(pdf_file)):
        if page_idx < start - 1:
            continue
        if page_idx >= end:
            break

        image_path = f"slide_{page_idx + 1}.jpg"
        page = pdf_file[page_idx]

        mat = pymupdf.Matrix(4, 4)  # zoom factor 2 in each direction
        rect = page.rect
        mp = (rect.tr + rect.br) / 2
        clip = pymupdf.Rect(rect.tl, mp)
        pix = page.get_pixmap(matrix=mat, clip=clip)
        pix.save(image_path)
   
        content = extract_slide_content(client, image_path, ctx)

        # check the format, if it doesn't have [Summary] then retry
        cnt = 0
        flag = "[Summary]" not in content
        while flag:
            time.sleep(1)
            print("retrying")
            content = extract_slide_content(client, image_path, ctx)
            flag = "[Summary]" not in content
            cnt += 1
            if cnt > 5:
                break
        if flag:
            logger.error(f"Failed to process slide {page_idx + 1}")
            continue

        ctx = content.split("[Summary]")[1].strip()
        content = content.split("[Summary]")[0].strip()
        
        # Create chunks from the content
        chunks = create_chunks(content, page_idx + 1)
        all_chunks.extend(chunks)
        
        # Remove the image file
        os.remove(image_path)
        
        logger.info(f"Processed slide {page_idx + 1}")
        time.sleep(0.1)
    pdf_file.save("output.pdf")
        
    # Save chunks to output JSON
    with open(output_json_path, 'w') as f:
        json.dump(all_chunks, f, indent=2)
    
    logger.info(f"Processed {len(all_chunks)} chunks saved to {output_json_path}")

if __name__ == "__main__":
    input_path = "/root/autodl-tmp/file2chunk/pdf_test/Lotus 424B3-20241121.pdf"
    start = 20 # Page start from 1
    end = 21
    process_slides(input_path, "1121ppt_test.json", start, end)