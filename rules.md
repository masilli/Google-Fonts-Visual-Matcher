# Project Overview
Google Fonts Visual Matcher: A locally-hosted tool that takes an image of text, processes it, and finds the closest Google Font alternative using visual embeddings and vector search.

# Tech Stack
- **Backend:** Python, FastAPI, Uvicorn (Local Server)
- **Machine Learning & Vision:** PyTorch, Torchvision (ResNet-50 for feature extraction)
- **Vector Search:** FAISS (CPU)
- **Image Processing:** Pillow (PIL), NumPy
- **Frontend (Local PoC):** Simple HTML/JS with Tailwind CSS served directly by FastAPI

# Architectural Rules
1. **Separation of Concerns:** Keep the indexing logic (`build_index.py`) strictly separated from the inference/API logic (`main.py`).
2. **Local Execution:** The system must run entirely on a local server. Do not rely on external cloud APIs for inference.
3. **Model Choice:** Use a pre-trained ResNet-50 (with the final classification layer removed) to generate 2048-dimensional embeddings. Do not attempt to train a new model from scratch.
4. **Vector Search:** Use `faiss.IndexFlatIP` with L2-normalized vectors to achieve Cosine Similarity matching.
5. **Image Preprocessing:** Always convert user-uploaded images to grayscale. Implement basic background detection (if background is dark and text is light, invert the image so text is always black on a white background).
6. **Code Style:** Use Python type hinting (`->`, `list`, `dict`), robust error handling (FastAPI `HTTPException`), and modular functions.
7. **Version Control Automation:** After every change, modification, or completed task, automatically stage, commit (with a clear, descriptive message), and push all changes to GitHub (`git push origin main`).


# Directory Structure
/fonts (directory for raw .ttf/.otf files)
build_index.py (script to render fonts and build FAISS index)
main.py (FastAPI server)
index.html (Local frontend interface)
requirements.txt (Dependencies)