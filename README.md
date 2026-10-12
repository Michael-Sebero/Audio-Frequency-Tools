## How to Install
```
git clone https://github.com/Michael-Sebero/Audio-Frequency-Tools
cd Audio-Frequency-Tools
pip install numpy scipy soundfile sounddevice

python3 audio-frequency-tools.py
```
Live playback also needs the PortAudio system library (`sudo pacman -S portaudio`,
`sudo apt-get install libportaudio2` or `brew install portaudio`). Without it,
you can still export sessions to files and analyse audio.

If packages are missing, the script offers to install them with the pip that belongs
to the running interpreter.

## Audio Frequency Generator
Plays or exports research-based presets in four suites (General, Attention & Cognition,
MindWar / BWR Manual, Alzheimer's / 40 Hz), or a manual configuration.

- **Delivery modes:** binaural (headphones), monaural beat, isochronic / click train,
  pure tone, infrasonic, high-carrier AM, parametric, surround.
- **Studied parameters:** each preset uses the tones, masking noise, timing and exposure
  of its cited study, and states how strong the evidence is.
- **Sessions:** continuous, timed, or pre-task induction. Sessions fade in and out,
  including when stopped with Ctrl+C.
- **Export:** write any session to WAV, FLAC, OGG or AIFF.
- **Embed mode:** apply amplitude modulation to an existing recording.

## Audio Analyser
`python3 audio-frequency-tools.py --detect FILE` reports the strongest tones in each
channel, the left/right offset (binaural beat) and any amplitude-modulation rhythm
(monaural beats, isochronic pulses). Use it to check what a "brainwave" track
actually contains.

## Self-test
`python3 audio-frequency-tools.py --selftest` renders every preset, confirms with the
analyser that it contains the intended frequencies, and checks for clicks and seams.

## Evidence
Read the [BWR Manual](BWR%20Manual) for an evidence-graded guide to each frequency band.
