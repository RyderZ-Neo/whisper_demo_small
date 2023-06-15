import streamlit as st
import time
from audio_recorder_streamlit import audio_recorder
import whisper
import numpy as np
from scipy.io.wavfile import read, write
import io
import ffmpeg
import os 
from io import BytesIO
import requests
import librosa.display
from matplotlib import pyplot as plt



HF_API_KEY= st.secrets["HF_API_KEY"]
API_URL = "https://api-inference.huggingface.co/models/valhalla/distilbart-mnli-12-3"
headers = {"Authorization": f"Bearer {HF_API_KEY}"}



st.set_page_config(page_title="You Speak I Write",page_icon=":ghost:", layout='wide', initial_sidebar_state="collapsed")

def plot_audio_transformations(y, sr):
    cols = [1, 1, 1]

    col1, col2, col3 = st.columns(cols)
    with col1:
        st.markdown(
            f"<h4 style='text-align: center; color: black;'>Mel Spectogram</h5>",
            unsafe_allow_html=True,
        )
        st.pyplot(plot_transformation(y, sr, "Original"))
    with col2:
        st.markdown(
            f"<h4 style='text-align: center; color: black;'>Wave plot </h5>",
            unsafe_allow_html=True,
        )
        st.pyplot(plot_wave(y, sr))
    with col3:
        st.markdown(
            f"<h4 style='text-align: center; color: black;'>Audio</h5>",
            unsafe_allow_html=True,
        )
        spacing()
        #st.audio(create_audio_player(y, sr))
        st.audio(audio_bytes, format="audio/wav")
def plot_transformation(y, sr, transformation_name):
    D = librosa.stft(y)  # STFT of y
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear', ax=ax)
    ax.set(title=transformation_name)
    fig.colorbar(img, ax=ax, format="%+2.f dB")

    return plt.gcf()
def plot_wave(y, sr):
    fig, ax = plt.subplots()
    img = librosa.display.waveshow(y, sr=sr, x_axis="time", ax=ax)
    return plt.gcf()
def spacing():
    st.markdown("<br></br>", unsafe_allow_html=True)

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

@st.cache_resource
def load_model(model_type):
   return whisper.load_model(model_type)

def load_audio(file: (str, bytes), sr: int = 16000):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: (str, bytes)
        The audio file to open or bytes of audio file

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    
    if isinstance(file, bytes):
        inp = file
        file = 'pipe:'
    else:
        inp = None
    
    try:
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(cmd="ffmpeg", capture_stdout=True, capture_stderr=True, input=inp)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

   
# --- Initialising SessionState ---
if "load_state" not in st.session_state:
     st.session_state.load_state = False
if "intent_state" not in st.session_state:
     st.session_state.intent_state = False

# model = whisper.load_model("small.en")
# #st.success("Whisper Model Loaded")

model = load_model("base")
st.success("Whisper Base Model Loaded")

st.title('⾳ ➡ ✍🏻 Transcriber')

with st.expander('About this App'):
	st.markdown('''
	This Streamlit app uses the whisper API to perform automatic speech recognition together with other useful features including: 
	- `Record` user's Audio from Mic, `Detect` Language of Speech and `transcription ` 
	- `Classify` context of Speech as Good , Bad or Neutral

	Libraries used:
	- `streamlit` - web framework for python scripts
	- `whisper` - OpenAI's whisper library providing loading models and inference

	''')

# Recording
with st.container():
    st.write("---")
#--HEADER--
st.header("Recording Studio (錄音) 🎤")
# Records 3 seconds in any case
audio_bytes = audio_recorder(
  #energy_threshold=(-1.0, 1.0),
  pause_threshold=3.0,
  text="Please Speak",
  recording_color="#e8b62c",
  neutral_color="#6aa36f",
  icon_name="microphone",
  icon_size="2x",
  sample_rate=16000
)
if audio_bytes is not None:
    audio_array = load_audio(audio_bytes)

if st.checkbox(":blue[Show Audio]"):
    plot_audio_transformations(audio_array,16000)


if st.checkbox(":violet[Detect Language]"):
        audio_array = whisper.pad_or_trim(audio_array)
        mel = whisper.log_mel_spectrogram(audio_array).to(model.device)
        _, probs = model.detect_language(mel)
        st.subheader(f"Detected language: {max(probs, key=probs.get)}") 

if st.button(":green[Transcribe Audio]")or st.session_state.load_state:
        st.session_state.load_state = True
        if audio_array is not None:  
         with st.spinner(text='In progress'):
            st.success("Transcribing Audio")
            transcription = model.transcribe(audio_array)
            st.markdown(transcription["text"])
            st.success("Transcription Complete")
        
if st.button(":green[Good 👏🏻,Bad 👎🏻 or Whatever!🤷]") or st.session_state.intent_state:
        st.session_state.intent_state = True
        with st.spinner("Just a moment"):
            st.success("An AI GURU Has Answered!")
            intent_inputs = transcription["text"]
            intent_labels = ['positive','negative','neutral']
            #st.text(intent_labels)
            payload = {
                "inputs": intent_inputs,
                "parameters": {"candidate_labels": intent_labels}
                        }
            intent_output = query(payload)
            st.write(intent_output)



