
import os
from os.path import exists, join, basename, splitext

git_repo_url = 'https://github.com/CorentinJ/Real-Time-Voice-Cloning.git'
project_name = splitext(basename(git_repo_url))[0]
if not exists(project_name):
  # clone and install
  !git clone -q --recursive {git_repo_url}
  # install dependencies
  !cd {project_name} && pip install -q -r requirements.txt
  !pip install -q gdown
  !apt-get install -qq libportaudio2
  !pip install -q https://github.com/tugstugi/dl-colab-notebooks/archive/colab_utils.zip

  # download pretrained model
  !cd {project_name} && wget https://github.com/blue-fish/Real-Time-Voice-Cloning/releases/download/v1.0/pretrained.zip && unzip -o pretrained.zip

import sys
sys.path.append(project_name)

from IPython.display import display, Audio, clear_output
from IPython.utils import io
import ipywidgets as widgets
import numpy as np
from dl_colab_notebooks.audio import record_audio, upload_audio

from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path

encoder.load_model(project_name / Path("encoder/saved_models/pretrained.pt"))
synthesizer = Synthesizer(project_name / Path("synthesizer/saved_models/pretrained/pretrained.pt"))
vocoder.load_model(project_name / Path("vocoder/saved_models/pretrained/pretrained.pt"))

#@title Record or Upload
#@markdown * Either record audio from microphone or upload audio from file (.mp3 or .wav) 

SAMPLE_RATE = 22050
record_or_upload = "Upload (.mp3 or .wav)" #@param ["Record", "Upload (.mp3 or .wav)"]
record_seconds =   29#@param {type:"number", min:1, max:10, step:1}

embedding = None
def _compute_embedding(audio):
  display(Audio(audio, rate=SAMPLE_RATE, autoplay=True))
  global embedding
  embedding = None
  embedding = encoder.embed_utterance(encoder.preprocess_wav(audio, SAMPLE_RATE))
def _record_audio(b):
  clear_output()
  audio = record_audio(record_seconds, sample_rate=SAMPLE_RATE)
  _compute_embedding(audio)
def _upload_audio(b):
  clear_output()
  audio = upload_audio(sample_rate=SAMPLE_RATE)
  _compute_embedding(audio)

if record_or_upload == "Record":
  button = widgets.Button(description="Record Your Voice")
  button.on_click(_record_audio)
  display(button)
else:
  #button = widgets.Button(description="Upload Voice File")
  #button.on_click(_upload_audio)
  _upload_audio("")

#@title Synthesize a text 
text = "when Alexander saw the breadth of his domain,he wept for there were no worlds to conquer" #@param {type:"string"}
print(embedding)
fp = open('sound.txt','w') 
fp.write(str(embedding))
def synthesize(embed, text):
  print("Synthesizing new audio...")
  #with io.capture_output() as captured:
  specs = synthesizer.synthesize_spectrograms([text], [embed])
  generated_wav = vocoder.infer_waveform(specs[0])
  generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
  clear_output()
  display(Audio(generated_wav, rate=synthesizer.sample_rate, autoplay=True))

if embedding is None:
  print("first record a voice or upload a voice file!")
else:
  synthesize(embedding, text)
