**There is some important enviroment variables u need to set**

```Linux Ubuntu Terminal 
export "GEMINI_API_KEY"="YOUR GEMINI API KEY"
export "HF_TOKEN"="YOU HUGGING FACE AUTHORIZATED ACCOUNT API TOKEN" 
```

```Windows cmd 
set "GEMINI_API_KEY"="YOUR GEMINI API KEY"
set "HF_TOKEN"="YOU HUGGING FACE AUTHORIZATED ACCOUNT API TOKEN" 
```
Here's an example of how you could write the instructions in the README file for your colleagues, guiding them on how to use the FastAPI service and explaining the recent changes related to resolving the Pydantic warning:

## Getting Started

Ensure you have the following installed:

- Python 3.10 or above
- Pip (Python package installer)
- Git

# Gemini Vacancy API (Branch: `gemini_only`)

This FastAPI application provides several endpoints for working with text-based vacancy generation, discrimination detection, feature extraction, and embeddings using the Gemini API models. The endpoints support job vacancy creation and filtering, as well as text similarity comparison using embeddings.

## Installation

To set up and run this project, follow the steps below:

### 1. Clone the repository

```bash
git clone https://github.com/goin2crazy/august_internchip.git
cd august_internchip
```

### 2. Set up a virtual environment (optional but recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install the required dependencies

Install the dependencies listed in the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

If you're running on a weak CPU or GPU and want a lightweight installation of PyTorch, use the following command:

#### Light PyTorch Installation for Weak CPU and GPU

```bash
pip install torch==2.0.1+cpu torchvision==0.15.2+cpu torchaudio==2.0.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

This version of PyTorch is optimized for CPU-only systems. If you have a CUDA-enabled GPU, you can install the appropriate GPU version by following the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

### 4. Run the application

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

You can now access the API at `http://127.0.0.1:8000`.

---

## API Endpoints

### 1. **Extract Features with Gemini**  
   **Endpoint**: `/extract_features_with_gemini`  
   **Method**: `POST`  
   **Description**: Extracts features from the input text using the Gemini feature extraction model.  
   **Request Body**:
   ```json
   {
     "text": "Sample text to extract features from"
   }
   ```
   **Response**: Returns the extracted features as a string.
   Example: 

   ```python
    {'result': '{"Core Responsibilities": ["N/A"],
              "Required Skills": ["N/A"],
              "Educational Requirements": ["N/A"],
              "Experience Level": ["N/A"],
              "Preferred Qualifications": ["N/A"],
              "Compensation and Benefits": ["N/A"]\}'}
  ```
  as str

### 2. **Direct Call Gemini Model**  
   **Endpoint**: `/call_gemini_directly`  
   **Method**: `POST`  
   **Description**: Calls the Gemini model to generate a response based on the input text.  
   **Request Body**:
   ```json
   {
     "text": "Sample text"
   }
   ```
   **Response**: Returns the model's output.
   Example: `{'result': [GENERATED ANSWER] }`

### 3. **Generate Vacancy Text**  
   **Endpoint**: `/call_gemini_write_vacancy/`  
   **Method**: `POST`  
   **Description**: Generates a vacancy text based on provided job features using the Gemini vacancy generation model.  
   **Request Body**:
   ```json
   {
     "job_features": {
       "title": "Job Title",
       "salary": "50000",
       "company": "Company Name",
       "experience": "2 years",
       "mode": "online",
       "skills": "Python"
     },
     "input_text": "Any additional input text"
   }
   ```
   **Response**: Returns the generated vacancy description like `{"result": "GENERATED VACANCY DESCRIPTION"}`

### 4. **Detect Discrimination in Vacancy Text**  
   **Endpoint**: `/detect_discrimination_with_gemini/`  
   **Method**: `POST`  
   **Description**: Detects if a vacancy text contains discriminatory content using the Gemini discrimination filter model.  
   **Request Body**:
   ```json
   {
     "text": "Vacancy text to check for discrimination"
   }
   ```
   **Response**: Returns a JSON object containing:
   - `reasoning`: A detailed reasoning about whether discrimination was detected.
   - `summarized_reasoning`: A short summary of the discrimination detection.
   - `answer`: Either `<warning>` if discrimination was found, or `<fine>` if the text is acceptable.
   - `warning`: True if answer is`<warning>` else False  

### 5. **Extract Embeddings**  
   **Endpoint**: `/extract_embeddings`  
   **Method**: `POST`  
   **Description**: Extracts embeddings from the input text for similarity comparison.  
   **Request Body**:
   ```json
   {
     "text": "Sample text to extract embeddings from"
   }
   ```
   **Response**: Returns the extracted embeddings in a list format.

### 6. **Compare Embeddings**  
   **Endpoint**: `/compare_embeddings`  
   **Method**: `POST`  
   **Description**: Compares two sets of embeddings using cosine similarity.  
   **Request Body**:
   ```json
   {
     "embedding1": [0.1, 0.2, 0.3],
     "embedding2": [0.4, 0.5, 0.6]
   }
   ```
   **Response**: Returns the cosine similarity score between the two embeddings.

---

## Project Structure

- **`texts_writer/`**: Contains configuration files and helper functions for writing and generating vacancy texts.
- **`remote_api_models/`**: Contains the Gemini API models used for feature extraction, vacancy generation, and discrimination detection.
- **`main.py`**: The FastAPI application, which exposes various endpoints to interact with the Gemini models.

---

## Contributing

Feel free to submit pull requests or issues if you find any bugs or want to suggest improvements.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
