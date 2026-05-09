import streamlit as st
from transformers import pipeline
from PIL import Image
from gtts import gTTS
import tempfile
import os

# -------------------------------------------------
# Page configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Kids Storytelling App",
    page_icon="📚",
    layout="centered"
)

# -------------------------------------------------
# Cached model loader
# -------------------------------------------------
@st.cache_resource
def load_caption_pipeline():
    """
    Load the Hugging Face image captioning pipeline.
    """
    return pipeline(
        task="image-to-text",
        model="Salesforce/blip-image-captioning-base"
    )

# -------------------------------------------------
# Helper functions
# -------------------------------------------------
def generate_image_caption(image):
    """
    Generate a caption from the uploaded image using Hugging Face pipeline.
    
    Args:
        image: PIL Image object
        
    Returns:
        str: Generated caption
    """
    caption_pipeline = load_caption_pipeline()
    results = caption_pipeline(image)

    if results and isinstance(results, list):
        return results[0]["generated_text"].strip()
    return "A wonderful picture for a children's story."

def limit_story_length(text, min_words=50, max_words=100):
    """
    Ensure the story length is between 50 and 100 words.
    
    Args:
        text (str): Story text
        min_words (int): Minimum word count
        max_words (int): Maximum word count
        
    Returns:
        str: Adjusted story text
    """
    words = text.split()

    if len(words) > max_words:
        text = " ".join(words[:max_words])

    if len(text.split()) < min_words:
        extra = (
            " They went home with happy hearts and promised to remember "
            "their lovely adventure forever."
        )
        text += extra

    return text

def generate_story_from_caption(caption):
    """
    Generate a kid-friendly story based on the image caption.
    
    Args:
        caption (str): Image caption
        
    Returns:
        str: Story text
    """
    story = (
        f"One sunny day, {caption} became part of a magical adventure. "
        f"A little child looked closely and smiled with wonder. "
        f"Soon, new friends appeared and everyone worked together kindly. "
        f"They explored, laughed, and discovered something special along the way. "
        f"At the end of the day, they learned that courage, friendship, and curiosity "
        f"can turn even an ordinary moment into a beautiful story."
    )

    return limit_story_length(story)

def text_to_speech(text):
    """
    Convert story text to speech and save as an MP3 file.
    
    Args:
        text (str): Story text
        
    Returns:
        str: Path to temporary MP3 file
    """
    tts = gTTS(text=text, lang="en")
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    return temp_file.name

# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------
st.title("📚 Kids Storytelling Application")
st.subheader("Upload a picture and create a fun story!")
st.write(
    "This app uses a Hugging Face image captioning model to understand your picture "
    "and then creates a short story for children aged 3–10."
)

uploaded_file = st.file_uploader(
    "Choose an image file",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Generate Story"):
        with st.spinner("Analyzing the image and creating a story..."):
            caption = generate_image_caption(image)
            story = generate_story_from_caption(caption)
            audio_path = text_to_speech(story)

        st.success("Story created successfully!")

        st.markdown("### 🖼 Image Caption")
        st.write(caption)

        st.markdown("### ✨ Generated Story")
        st.write(story)

        word_count = len(story.split())
        st.write(f"Word count: {word_count}")

        st.markdown("### 🔊 Story Audio")
        with open(audio_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format="audio/mp3")

            st.download_button(
                label="Download Story Audio",
                data=audio_bytes,
                file_name="story_audio.mp3",
                mime="audio/mpeg"
            )

        if os.path.exists(audio_path):
            os.remove(audio_path)
else:
    st.info("Please upload an image to begin.")
