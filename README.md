# Hope Speech Detection in Thanglish (YouTube Comments)

Hope Speech Detection on Tamil-English code-mixed text (**Thanglish**) using XLM-RoBERTa, enhanced with hybrid preprocessing (dictionary-based normalization + Levenshtein fuzzy matching). The pipeline is designed to handle real-world social media text variations, spelling mistakes, and Tamil-English code-mixing — and is deployed as a live API that classifies hope speech directly from YouTube video comments.

---

## Features

**Hybrid Preprocessing**
- Exact dictionary normalization
- Levenshtein distance–based fuzzy matching
- Handles stretched words and Tamil-English code-mixing

**Model**
- XLM-RoBERTa (base, ~270M parameters)
- Supports 100+ languages (Tamil + English included)
- Optimized training with early stopping

**Dataset**
- ~14k training samples, ~1.7k validation, ~1.7k test
- Labels: `Hope_speech`, `Non_hope_speech`

**Results**
- Test Accuracy: ~67%
- Normalization Coverage: ~3.6% of words normalized
- Strong multilingual performance on noisy, code-mixed social media text

**Advantages over DistilBERT**
- Better multilingual understanding
- Robust to spelling variations
- Stronger handling of Tamil-English code-mixing
- Contextual embeddings suited to noisy social media text

---

## Live Deployment

The model is deployed as a REST API on Hugging Face Spaces, with interactive Swagger (OpenAPI) documentation available at the Space's `/docs` endpoint.

**Base URL:** `https://tamilselvan0709-hope-speech-classifier.hf.space/docs#/`


### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check — confirms the API is running |
| `POST` | `/analyze` | Extracts comments from a YouTube video and classifies each as `Hope_speech` or `Non_hope_speech` |

---

## How to Run

### Option 1 — Using the Swagger UI (no setup required)

1. Open `https://tamilselvan0709-hope-speech-classifier.hf.space/docs#/` in your browser.
2. Expand the **`POST /analyze`** endpoint.
3. Click **"Try it out"**.
4. In the request body, replace the placeholder `"string"` with a real YouTube video URL:
   ```json
   {
     "url": "https://www.youtube.com/watch?v=VIDEO_ID"
   }
   ```
5. Click **"Execute"**.
6. The API will:
   - Extract comments from the given YouTube video
   - Run each comment through the hybrid preprocessing pipeline
   - Classify each comment using the trained XLM-RoBERTa model
   - Return the labeled results in the response body

### Option 2 — Using `curl`

```bash
curl -X POST "https://tamilselvan0709-hope-speech-classifier.hf.space/docs#/" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=VIDEO_ID"}'
```

### Option 3 — Using Python

```python
import requests

response = requests.post(
    "https://tamilselvan0709-hope-speech-classifier.hf.space/docs#/",
    json={"url": "https://www.youtube.com/watch?v=VIDEO_ID"}
)

results = response.json()
print(results)
```

---

## Example Output

Each extracted comment is returned with its original text, the normalized/preprocessed text, the predicted label, and the model's confidence score:

```json
{
  "original_text": "Semma video! This gives me hope and courage to achieve my dreams",
  "processed_text": "awesome video! This gives me hope and courage to achieve my dreams",
  "predicted_class": "Hope_speech",
  "confidence": 0.87
}
```

For a video with multiple comments, the API returns a list of these objects — one per extracted comment.

---

## Tech Stack

- **Model:** XLM-RoBERTa (base)
- **Preprocessing:** Dictionary normalization + Levenshtein fuzzy matching
- **Backend:** FastAPI (OpenAPI 3.1 / Swagger-documented)
- **Deployment:** Hugging Face Spaces

---

## Notes

- Comment extraction depends on the video's comments being publicly accessible; videos with comments disabled will return no results.
- Classification quality may vary on extremely short comments, emoji-only comments, or comments in scripts outside Tamil/English code-mixing patterns.
