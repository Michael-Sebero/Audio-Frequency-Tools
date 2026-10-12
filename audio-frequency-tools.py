"""
Audio Frequency Tools
=====================================
Brainwave-entrainment tone generator (binaural, monaural, isochronic, pure
tone, infrasonic, high-carrier AM, parametric and surround delivery modes,
plus an embed mode for existing recordings) and an analyser that measures
carrier tones, binaural offsets and amplitude-modulation rates in audio files.

Every preset states the study it comes from, the stimulus that study actually
used, and how strong the evidence is.  AUDIT.md lists the corrections made
against the source literature.

Usage
  python3 audio-frequency-tools.py                  interactive menu
  python3 audio-frequency-tools.py --detect FILE    analyse an audio file
  python3 audio-frequency-tools.py --selftest       render every preset and
                                                     verify it with the analyser
Requirements
  numpy, scipy, soundfile  (+ sounddevice and PortAudio for live playback)
"""

import argparse
import importlib
import importlib.util
import math
import os
import queue
import signal
import subprocess
import sys
import tempfile
import textwrap
import threading
import time

# ---------------------------------------------------------------------------
#  DEPENDENCIES
# ---------------------------------------------------------------------------

_REQUIRED_DEPS = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("soundfile", "soundfile"),
    ("sounddevice", "sounddevice"),   # only needed for live playback
]


def _is_installed(module_name: str) -> bool:
    """Check whether a module can be found without importing it.  Importing
    sounddevice raises OSError when the PortAudio system library is missing,
    which pip cannot fix, so that case is handled at playback time instead."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def _pip_install(packages) -> bool:
    """Install packages with the pip belonging to *this* interpreter."""
    cmd = [sys.executable, "-m", "pip", "install", "--break-system-packages", *packages]
    print("  Running: " + " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  pip failed: {e}")
        return False


def _ensure_deps():
    missing = [pkg for mod, pkg in _REQUIRED_DEPS if not _is_installed(mod)]
    if not missing:
        return
    print(f"Missing Python packages: {', '.join(missing)}")
    installed = False
    if sys.stdin.isatty():
        ans = input("Install them now with pip for this interpreter? [y/N]: ").strip().lower()
        if ans in ("y", "yes"):
            installed = _pip_install(missing)
            importlib.invalidate_caches()
    if not installed:
        print(f"Install manually:  {sys.executable} -m pip install {' '.join(missing)}")
        if any(p != "sounddevice" for p in missing):
            sys.exit(1)
        print("Continuing without live playback (export and analysis still work).\n")


_ensure_deps()

import numpy as np
import soundfile as sf
from scipy import fft as spfft
from scipy import signal as sps


def _load_sounddevice():
    """Import sounddevice lazily so export, analysis and the self-test work on
    machines without PortAudio."""
    try:
        import sounddevice as sd
        return sd
    except ImportError:
        print("  Live playback needs the 'sounddevice' package:")
        print(f"    {sys.executable} -m pip install sounddevice")
    except OSError as e:
        print(f"  Live playback unavailable: {e}")
        print("  The 'sounddevice' package is installed, but the PortAudio system")
        print("  library is missing.  pip cannot install it for you.")
        print("    Arch          : sudo pacman -S portaudio")
        print("    Debian/Ubuntu : sudo apt-get install libportaudio2")
        print("    macOS         : brew install portaudio")
    print("  You can still export the session to an audio file instead.")
    return None

# ---------------------------------------------------------------------------
#  CONSTANTS
# ---------------------------------------------------------------------------

SR_DEFAULT           = 44100
BLOCK_SIZE           = 4096
PREFILL              = 4
DTYPE                = "float32"
FADE_S               = 4.0      # fade in / scheduled fade out
STOP_FADE_S          = 1.5      # fade out after Ctrl+C
CARRIER_OPTIMAL      = 400.0    # Goodin et al. (2012), via Ingendoh et al. (2023)
SUBLIMINAL_CARRIER   = 18500.0
PARAMETRIC_BASE      = 40000.0
BINAURAL_LIMIT       = 30.0     # beat percept limit, Perrott & Nelson (1969)
BINAURAL_CARRIER_MAX = 1000.0   # Licklider et al. (1950); Oster (1973)
NOISE_RMS            = 0.25     # fixed masking-noise level before blending
DEFAULT_NOISE_LVL    = 0.25

MODES = {
    "binaural":   "two tones, one per ear; headphones required",
    "monaural":   "two tones mixed in both ears; physical beat, speakers OK",
    "isochronic": "carrier switched on/off at the beat rate; speakers OK",
    "mono":       "single pure tone at the given frequency",
    "infrasonic": "pure tone below 20 Hz; needs a subwoofer",
    "hf_am":      "high-frequency carrier, amplitude-modulated (was 'subliminal')",
    "parametric": "two ultrasonic tones; needs a parametric speaker array",
    "surround":   "isochronic tone on every output channel",
}

# ---------------------------------------------------------------------------
#  PRESET DATA
#
#  ref_s / ref_src : exposure used in the cited study.  No study establishes a
#                    minimum effective dose; Garcia-Argibay et al. (2018) found
#                    larger effects with longer exposure (b = 0.01 per minute).
#  timing          : when the cited study played the audio relative to the task.
#  pre_s           : default pre-task induction for "before_and_during" presets.
#  noise/noise_lvl : masking noise used in the cited study (offered as default).
#  pulse_ms        : isochronic pulse width; overrides duty when set.
# ---------------------------------------------------------------------------

_PRESET_DEFAULTS = dict(
    duty=0.5, pulse_ms=None, timing="during", ref_s=None, ref_src="", pre_s=None,
    noise=None, noise_lvl=0.0, evidence="", desc="", detail="",
)


def preset(**kw):
    return {**_PRESET_DEFAULTS, **kw}


SUITE_GENERAL = [
    preset(
        label    = "Delta  1.5 Hz  --  Anxiety reduction before a stressful event",
        mode     = "binaural", beat_hz=1.5, carrier_hz=400.0,
        evidence = "Moderate (anxiety) / none (sleep)",
        timing   = "before", ref_s=1800, ref_src="Padmanabhan 2005; Wahbeh 2007b",
        noise    = "pink", noise_lvl=DEFAULT_NOISE_LVL,
        desc     = """\
    Delta (0.5-4 Hz) dominates deep NREM sleep.  Delta-range beats heard for
    30 min BEFORE a stressful event reduced anxiety (Padmanabhan 2005;
    Wahbeh 2007b).  No study in the provided literature tested sleep.
    Headphones required.  Evidence: Moderate (anxiety) / none (sleep).""",
        detail   = """\
    Delta oscillations (0.5-4 Hz) are the dominant rhythm of deep (N3) sleep.
    Waking delta does not reproduce the restorative effects of sleep.

    WHAT THE STUDIES SHOW
      Padmanabhan et al. (2005): delta-range binaural beats for 30 min before
        surgery cut anxiety scores by 26.3%, vs 11.1% for a sham recording
        (Chaieb et al. 2015 review).
      Wahbeh et al. (2007b): delta beats in pink noise, 30 min a day for 60
        days, lowered trait anxiety (uncontrolled pilot, n = 8).
      Garcia-Argibay et al. (2018) meta-analysis, anxiety subgroup:
        g = 0.69 [0.39, 0.99], 5 effect sizes, no heterogeneity.
      Lane et al. (1998): 1.5/4 Hz beats DURING a 30-min vigilance task gave
        worse vigilance and more negative mood than 16/24 Hz beats.  Do not
        use delta when you need to stay alert.

    EEG: Gao et al. (2014) found no delta power increase after 5 min of a
    1 Hz beat (Ingendoh et al. 2023).  Below ~3 Hz the beat is heard as a
    tone rotating between the ears rather than a pulsation.

    Delivery: binaural beat, 400 Hz carrier (Goodin et al. 2012), pink noise
    as in Wahbeh 2007b.  Headphones required.
    Studied exposure: 30 min, before the event.""",
    ),
    preset(
        label    = "Theta  6 Hz  --  Relaxation / anxiety reduction (not for study)",
        mode     = "binaural", beat_hz=6.0, carrier_hz=250.0,
        evidence = "Mixed (EEG) / Moderate (anxiety)",
        timing   = "before", ref_s=600, ref_src="Isik 2017; Jirakittayakorn & Wongsawat 2017",
        desc     = """\
    Theta (4-7 Hz) is prominent in drowsiness, hypnagogia and meditation.
    10 min of theta beats before dental surgery reduced anxiety (Isik 2017).
    Theta beats IMPAIRED word recall (Garcia-Argibay 2017; Wahbeh 2007a), so
    do not use while studying.  Headphones required.
    Evidence: Mixed (EEG) / Moderate (anxiety).""",
        detail   = """\
    Theta oscillations (4-7 Hz) are prominent during drowsiness, the
    hypnagogic state and meditation.  Frontal-midline theta also accompanies
    focused internal attention.

    EEG EVIDENCE (Ingendoh et al. 2023 systematic review)
      For:     Jirakittayakorn & Wongsawat (2017, Front Neurosci): a 6 Hz beat
               on a 250 Hz carrier raised theta power at almost all electrodes
               over a 30-min session.  Seifi Ala et al. (2018): relative (not
               absolute) theta power rose at parietal and temporal sites.
               Reported onset: 6-10 min of listening.
      Against: Gao 2014, Lopez-Caballero 2017, Crespo 2013, da Silva Junior
               2019, Solca 2016 and Wahbeh 2007 found no theta entrainment.

    BEHAVIOUR
      Anxiety: Isik et al. (2017), 10 min before dental surgery, g = 0.73.
      Memory:  5 Hz beats lowered free recall and recognition (Garcia-Argibay
               2017).  7 Hz for 30 min lowered verbal recall (Wahbeh 2007a).
               Ortiz et al. (2008, 5 Hz, twice a day for 15 days) is the one
               positive exception.

    Delivery: binaural beat, 250 Hz carrier (Jirakittayakorn & Wongsawat
    protocol).  Headphones required.  Use at rest, before rather than during
    demanding cognitive work.  Studied exposure: 10 min (Isik) to 30 min.""",
    ),
    preset(
        label    = "Schumann  7.83 Hz  --  Earth-ionosphere resonance (no direct evidence)",
        mode     = "binaural", beat_hz=7.83, carrier_hz=400.0,
        evidence = "None found (Schumann-specific)",
        desc     = """\
    7.83 Hz is the fundamental electromagnetic resonance of the cavity
    between the Earth and the ionosphere, at the theta/alpha boundary.  No
    study in the provided literature tested 7.83 Hz audio.  Any effect would
    be that of a generic theta/alpha beat.  Headphones required.""",
        detail   = """\
    7.83 Hz is the fundamental resonance of the electromagnetic cavity
    between the Earth's surface and the ionosphere.  It is an electromagnetic
    phenomenon: an acoustic beat at the same number of cycles per second has
    no physical connection to it.

    No study in the provided documents tested 7.83 Hz beats.  The nearest
    evidence is for theta/delta beats in general, which consistently reduced
    anxiety (Chaieb et al. 2015; Garcia-Argibay et al. 2018).

    Delivery: binaural beat, 400 Hz carrier.  Headphones required.
    Evidence: none found for 7.83 Hz specifically.""",
    ),
    preset(
        label    = "Alpha  10 Hz  --  Relaxed wakefulness / stress reduction",
        mode     = "binaural", beat_hz=10.0, carrier_hz=250.0,
        evidence = "Mixed (EEG) / Limited (anxiety)",
        ref_s    = 300, ref_src="Kim 2023",
        desc     = """\
    Alpha (8-12 Hz) dominates relaxed, eyes-closed wakefulness.  Kim et al.
    (2023): 5 min of a 10 Hz beat on a 250 Hz carrier raised absolute alpha
    power everywhere except the temporal area.  Vernon 2014, Gao 2014 and
    others found no alpha entrainment.  Headphones required.
    Evidence: Mixed (EEG) / Limited (anxiety).""",
        detail   = """\
    Alpha oscillations (8-12 Hz) are the dominant rhythm during relaxed,
    wakeful rest.  Higher alpha is a correlate of reduced cortical arousal,
    not a direct cause of calm.

    EEG EVIDENCE (Ingendoh et al. 2023)
      For:     Kim et al. (2023): 10 Hz beat, 250 Hz carrier, 5 min; absolute
               alpha rose over frontal, central, parietal and occipital areas.
               Ioannou et al. (2015): alpha power rose during alpha beats.
      Against: Vernon 2014 (1-min epochs), Gao 2014, Lopez-Caballero 2017 and
               Solca 2016 found no alpha entrainment.

    BEHAVIOUR
      Weiland et al. (2011): nature sounds with an embedded 10 Hz beat for
        20 min lowered anxiety in emergency-department patients (Chaieb 2015).
      Reedijk et al. (2015): alpha beats, attentional blink, g = 0.43.
      Kraus & Porubanova (2015): 9.55 Hz beats improved working memory (see
        the Attention & Cognition suite).

    Delivery: binaural beat, 250 Hz carrier to replicate Kim et al. (2023).
    Headphones required.  Studied exposure: 5 min (Kim) to 20 min (Weiland).""",
    ),
    preset(
        label    = "15 Hz  --  BWR Manual 'discomfort' claim (not supported)",
        mode     = "monaural", beat_hz=15.0, carrier_hz=400.0,
        evidence = "Claim unsupported",
        desc     = """\
    The BWR Manual lists 15 Hz as causing headache and nausea.  No provided
    study reports this.  15 Hz binaural beats were used in an ADHD trial
    (Malandrone 2022), and 16 Hz beats improved vigilance and mood (Lane
    1998).  Played as a monaural beat (400 + 415 Hz) so it works on speakers.""",
        detail   = """\
    BWR MANUAL CLAIM: "15 Hz -> headache & nausea ... best format: mono".
    None of the provided studies supports this claim, and two contradict it:
      Malandrone et al. (2022): 15 Hz binaural beats (400/415 Hz) during study
        sessions in adult ADHD improved subjective study performance.
      Lane et al. (1998): 16/24 Hz binaural beats produced LESS negative mood
        than 1.5/4 Hz beats.
    The discomfort association comes from photosensitivity to flashing light,
    which is most provocative at roughly 15-20 flashes per second.  That is a
    visual effect.  It does not carry over to sound.

    FORMAT CORRECTION: the previous version played a raw 15 Hz sine wave.
    15 Hz is infrasonic: headphones and laptop speakers barely reproduce it,
    and at audible levels it is felt rather than heard.  The BWR Manual's
    "mono = speakers" intent is served by a monaural beat: both tones mixed
    into each channel, so the 15 Hz beat physically exists in the sound and
    is audible on any speaker (Chaieb et al. 2015).

    Delivery: monaural beat, 400 + 415 Hz in both channels.
    Speakers or headphones.""",
    ),
    preset(
        label    = "Low Beta  16 Hz  --  Vigilance / sustained attention (Lane 1998)",
        mode     = "binaural", beat_hz=16.0, carrier_hz=400.0,
        evidence = "Inconclusive (EEG) / Limited (behaviour)",
        timing   = "during", ref_s=1800, ref_src="Lane 1998",
        noise    = "pink", noise_lvl=DEFAULT_NOISE_LVL,
        desc     = """\
    Lane et al. (1998): 16/24 Hz beats in pink noise, played DURING a 30-min
    vigilance task, gave more correct detections, fewer false alarms and
    less negative mood than 1.5/4 Hz beats.  No beta-band EEG study found
    beta entrainment (Ingendoh 2023).  Headphones required.
    Evidence: Inconclusive (EEG) / Limited (behaviour).""",
        detail   = """\
    The beta band (13-30 Hz) accompanies focused attention and deliberate
    information processing.

    BEHAVIOUR
      Lane et al. (1998), n = 29, crossover: beta beats (16 and 24 Hz) in pink
        noise during a 30-min visual vigilance task beat theta/delta beats
        (1.5 and 4 Hz) on hits, false alarms and mood.  The comparison was
        theta/delta beats, not silence.
      A 4-min study of 7 and 16 Hz beats found no vigilance effect (Chaieb
        et al. 2015).

    EEG: none of the beta-band studies in Ingendoh et al. (2023) found beta
    entrainment (Gao 2014; Vernon 2014; Lopez-Caballero 2017; Crespo 2013;
    Corona-Gonzalez 2021).  Goodin et al. (2012) found no spectral change for
    16 Hz beats.  Behavioural effects, where present, may work through
    another mechanism, such as noradrenergic arousal (Hommel et al. 2016).

    Delivery: binaural beat, 400 Hz carrier (Lane's carriers are not given in
    the provided documents), pink noise as in Lane.  Headphones required.
    Studied exposure: 30 min, during the task.""",
    ),
    preset(
        label    = "High Beta  25 Hz  --  Heightened arousal  (caution)",
        mode     = "binaural", beat_hz=25.0, carrier_hz=400.0,
        evidence = "Inconclusive",
        desc     = """\
    High-beta (20-30 Hz) EEG accompanies heightened arousal and, in excess,
    anxiety.  That is a correlation, not evidence that 25 Hz beats cause
    anxiety.  Lane (1998) included 24 Hz in its beneficial beta condition.
    No EEG entrainment shown at this band.  Headphones required.""",
        detail   = """\
    High-beta oscillations (20-30 Hz) accompany heightened cortical arousal.
    In stress-prone people, excess high beta correlates with anxiety and
    rumination.  That association runs from mental state to EEG and does not
    show that a 25 Hz beat induces anxiety.

    Lane et al. (1998) used 16 and 24 Hz beats together as its beta
    condition, which improved vigilance and mood relative to theta/delta.
    No study in the provided documents found beta-band EEG entrainment
    (Ingendoh et al. 2023).

    Delivery: binaural beat, 400 Hz carrier.  Headphones required.
    Evidence: Inconclusive.""",
    ),
    preset(
        label    = "Gamma  40 Hz  --  Gamma-band response (Jirakittayakorn 2017)",
        mode     = "binaural", beat_hz=40.0, carrier_hz=250.0,
        evidence = "Mixed (EEG)",
        timing   = "before_and_during", ref_s=900, pre_s=900,
        ref_src  = "Jirakittayakorn & Wongsawat 2017 (Int J Psychophysiol)",
        desc     = """\
    Jirakittayakorn & Wongsawat (2017, Int J Psychophysiol): a 40 Hz beat on
    a 250 Hz carrier raised gamma power over temporal, frontal and central
    sites after 15 min of a 30-min session.  Engelbregt 2021 and
    Lopez-Caballero 2017 found no gamma increase.  Headphones required.
    Evidence: Mixed (EEG).  Attention studies: Attention & Cognition suite.""",
        detail   = """\
    Gamma oscillations (30-80 Hz) are linked to perceptual binding, working
    memory and cross-regional coordination.  Gamma deficits are documented in
    Alzheimer's disease and schizophrenia.

    EEG EVIDENCE
      For:     Jirakittayakorn & Wongsawat (2017, Int J Psychophysiol), n = 47:
               40 Hz beat, 250 Hz carrier, 30 min; gamma rose over temporal,
               frontal and central regions from 15 min.  Schwarz & Taylor
               (2005) and Draganova (2008) measured 40 Hz steady-state
               responses (ASSR) to 40 Hz binaural beats.
      Against: Engelbregt et al. (2021, 440/480 Hz) and Lopez-Caballero &
               Escera (2017) found no gamma power increase.

    CORRECTION: the previous version played this as an isochronic tone
    because "40 Hz exceeds the ~30 Hz binaural limit", while citing EEG
    results that came from binaural beats.  Perrott & Nelson (1969) put the
    limit for HEARING a binaural beat near 30 Hz (Licklider 1950: ~35 Hz).
    That is not a limit on brain responses.  40 Hz binaural beats evoke
    measurable responses (Schwarz & Taylor 2005; Orozco Perez 2020), and
    every gamma study cited in this tool used binaural delivery.

    Delivery: binaural beat, 250 Hz carrier (Jirakittayakorn & Wongsawat).
    Headphones required.  Studied exposure: 15 min (effect onset).""",
    ),
]

SUITE_ATTENTION = [
    preset(
        label    = "40 Hz Attention BB  --  Engelbregt 2021 (440/480 Hz binaural)",
        mode     = "binaural", beat_hz=40.0, carrier_hz=440.0,
        evidence = "Moderate (single crossover study, n = 24)",
        timing   = "during", ref_s=300, ref_src="Engelbregt 2021",
        desc     = """\
    Engelbregt et al. (2021): a 40 Hz binaural beat (440 Hz one ear, 480 Hz
    the other) played for 5 min DURING an Eriksen Flanker task produced fewer
    errors than pink noise or monaural beats.  Reaction time did not change.
    Monaural beats at the same tones INCREASED errors.  Headphones required.
    Evidence: Moderate (single crossover study, n = 24).""",
        detail   = """\
    Source: Engelbregt H, Barmentlo M, Keeser D, Pogarell O, Deijen JB (2021).
    Effects of binaural and monaural beat stimulation on attention and EEG.
    Experimental Brain Research 239:2781-2791.  doi:10.1007/s00221-021-06155-z

    DESIGN: 25 psychology students (24 analysed), within-subject crossover.
    Pink noise (PN), monaural beats (MB) and binaural beats (BB), 5 min each
    in random order, while performing the Flanker task, with EEG recorded.
    BB: 440 Hz in one ear and 480 Hz in the other (this tool: L 440, R 480).
    MB: both tones in both ears.

    RESULTS (Flanker false responses; one-tailed tests)
      BB < PN : F(1,21) = 3.486,  p = .038, partial eta^2 = .142
      MB > PN : F(1,21) = 18.711, p < .001, partial eta^2 = .471
      BB < MB : t(23) = 1.78,     p = .044, partial eta^2 = .122
      Reaction time: no difference between conditions.

    EEG: no consistent 40 or 45 Hz power increase with BB.  The only 40 Hz
    increase (electrode Fz) occurred with MONAURAL beats.  BB showed higher
    45 Hz power than MB at T5 only.  The authors attribute the attention
    benefit to something other than entrainment, possibly noradrenergic
    dynamics (Hommel et al. 2016).

    LIMITATIONS: small student sample; 5-min exposure, which the authors
    call too short; 440/480 Hz lies above the 100-400 Hz range in which
    binaural beats are best perceived (Licklider et al. 1950).

    Delivery: binaural beat.  Headphones required.  Mode matters: do not
    substitute monaural beats.  Studied exposure: 5 min, during the task.""",
    ),
    preset(
        label    = "15 Hz ADHD Focus BB  --  Malandrone 2022 (400/415 Hz binaural)",
        mode     = "binaural", beat_hz=15.0, carrier_hz=400.0,
        evidence = "Limited (pilot RCT, conference abstract)",
        timing   = "during",
        desc     = """\
    Malandrone et al. (2022) pilot RCT, adult ADHD: 400 Hz left / 415 Hz
    right during self-directed study sessions.  Subjective study performance
    improved in the beat group only, with a between-group difference at the
    last visit.  ADHD-RS, mind-wandering and SART did not change.
    Headphones required.  Evidence: Limited.""",
        detail   = """\
    Source: Malandrone F, Spadotto M, Boero M, Bracco IF, Oliva F (2022).
    A pilot add-on randomized-controlled trial evaluating the effect of
    binaural beats on study performance, mind-wandering, and core symptoms
    of adult ADHD patients.  European Psychiatry 65(S1):S274, EPP0446.
    doi:10.1192/j.eurpsy.2022.701

    DESIGN: university students on medication for adult ADHD.  Intervention:
    15 Hz beat (415 Hz right ear, 400 Hz left ear).  Placebo: the same track
    with 400 Hz in both ears (no beat).  Baseline plus two fortnightly
    follow-ups.  Self-administered during individual study sessions.

    RESULTS
      Subjective studying performance (SSP): improved from baseline to last
        observation in the beat group only (mean difference 2.7, p < .001);
        significant between-group difference at the last observation.
      ADHD-RS-5, MEWS (mind-wandering), SART (sustained attention): no
        significant change.

    INTERPRETATION: the gain was in self-rated study experience, not in
    objective attention.  Kennel et al. (2010) saw the same pattern in
    children with ADHD: subjectively fewer problems, no objective change
    (Chaieb et al. 2015).  Session length was not reported.

    LIMITATIONS: pilot, conference abstract, small sample, no EEG,
    medication confound.

    Delivery: binaural beat.  Headphones required.  Play during study.""",
    ),
    preset(
        label    = "Alpha  9.55 Hz  --  Working memory (Kraus & Porubanova 2015)",
        mode     = "binaural", beat_hz=9.55, carrier_hz=400.0,
        evidence = "Moderate (single study)",
        timing   = "before", ref_s=720, ref_src="Kraus & Porubanova 2015",
        noise    = "pink", noise_lvl=DEFAULT_NOISE_LVL,
        desc     = """\
    Kraus & Porubanova (2015): 9.55 Hz alpha beats in pink noise for 12 min
    BEFORE an automated OSPAN task improved working-memory capacity compared
    with a sea-sounds control (g = 0.68).  Use the pre-task induction option.
    Headphones required.  Evidence: Moderate (single study).""",
        detail   = """\
    Source: Kraus J, Porubanova M (2015).  The effect of binaural beats on
    working memory capacity.  Studia Psychologica 57(2):135.
    Garcia-Argibay et al. (2018) meta-analysis, Table 1 ID 22: g = 0.681.

    DESIGN (Garcia-Argibay 2018, Tables 1-2): RCT, 20 per group; 12 min of
    alpha beats masked with pink noise BEFORE the automated operation-span
    (OSPAN) task.  Control group: sea sounds.  The 9.55 Hz figure is from
    Engelbregt et al. (2021).

    CORRECTION: the previous version said the study used sea-sound masking
    DURING the task.  Sea sounds were the control condition, the beats were
    masked with pink noise, and exposure came before the task.

    TIMING: Garcia-Argibay et al. (2018) meta-regression found larger effects
    for exposure before (b = 0.46, p = .006) or before and during (b = 0.53,
    p = .002) the task than for exposure during it only.

    Delivery: binaural beat, 400 Hz carrier (the original carrier is not
    given in the provided documents), pink noise.  Headphones required.
    Studied exposure: 12 min, then begin the task.""",
    ),
    preset(
        label    = "Beta  20 Hz  --  Long-term memory encoding (Garcia-Argibay 2017)",
        mode     = "binaural", beat_hz=20.0, carrier_hz=400.0,
        evidence = "Moderate (single RCT, n = 32)",
        timing   = "before_and_during", ref_s=1020, ref_src="Garcia-Argibay 2017", pre_s=300,
        desc     = """\
    Garcia-Argibay et al. (2017): unmasked 20 Hz beta beats for 17 min,
    before and during word-list encoding, improved free recall and
    recognition sensitivity (d') compared with white noise.  5 Hz theta beats
    made both worse.  Headphones required.
    Evidence: Moderate (single RCT, n = 32).""",
        detail   = """\
    Source: Garcia-Argibay M, Santed MA, Reales JM (2019).  Binaural auditory
    beats affect long-term memory.  Psychological Research 83:1124-1136
    (online 2017).  doi:10.1007/s00426-017-0959-2
    Garcia-Argibay et al. (2018) meta-analysis, Table 1 IDs 13-16.

    DESIGN: 32 participants; 20 Hz (beta) or 5 Hz (theta) binaural beats, or
    white noise as the control, for 17 min before and during encoding,
    followed by free recall and recognition tests.

    RESULTS (Hedges' g vs white noise)
      Beta:  g = 0.91 and 1.50   (free recall and recognition d')
      Theta: g = -0.82 and -0.53 (theta IMPAIRED both)
      The meta-analysis tables do not unambiguously match each g to a task.
    The direction of the effect depends on the beat frequency.

    CORRECTIONS: the previous version used a 13 Hz beat and said the beats
    were masked with white noise.  The study used 20 Hz.  White noise was
    the control condition, and the beats were unmasked (Garcia-Argibay 2018,
    Table 2).

    CONTEXT: the memory subgroup of the meta-analysis was the weakest and
    most heterogeneous, g = 0.26 [0.02, 0.50], I^2 = 69%.

    Delivery: binaural beat, 400 Hz carrier (the carrier is not given in the
    provided documents).  Headphones required.
    Studied exposure: 17 min, starting before and continuing during study.""",
    ),
    preset(
        label    = "Gamma  40 Hz  --  Attentional focus (Colzato 2017, 340 Hz carrier)",
        mode     = "binaural", beat_hz=40.0, carrier_hz=340.0,
        evidence = "Moderate (attention subgroup g = 0.58)",
        timing   = "before_and_during", ref_s=180, ref_src="Colzato 2017", pre_s=180,
        noise    = "white", noise_lvl=DEFAULT_NOISE_LVL,
        desc     = """\
    Colzato et al. (2017): 40 Hz binaural beats on a 340 Hz carrier, in white
    noise, for 3 min before and during a global-local task produced a
    narrower attentional focus than a constant 340 Hz tone (g = 0.69).
    Hommel (2016) and Reedijk (2015) used the same carrier.
    Headphones required.  Evidence: Moderate.""",
        detail   = """\
    Sources:
    Colzato LS, Barone H, Sellaro R, Hommel B (2017).  More attentional
      focusing through binaural beats: evidence from the global-local task.
      Psychological Research 81:271-277.  doi:10.1007/s00426-015-0727-0
    Hommel B, Sellaro R, Fischer R, Borg S, Colzato LS (2016).  High-frequency
      binaural beats increase cognitive flexibility.  Front Psychol 7:1287.
    Reedijk SA, Bolders A, Colzato LS, Hommel B (2015).  Eliminating the
      attentional blink through binaural beats.  Front Psychiatry 6:82.

    COLZATO 2017: 36 healthy adults.  40 Hz beats "based on a 340 Hz carrier
    tone", embedded in white noise, for 3 min before and during the task.
    Control: constant 340 Hz tone.  The global-precedence effect shrank
    (g = 0.69).  HOMMEL 2016: 40 Hz beats increased cognitive flexibility
    (g = 0.67).  REEDIJK 2015: 40 Hz beats eliminated the attentional blink
    (g = 0.83), but only in people with low spontaneous eye-blink rates.

    Garcia-Argibay et al. (2018), attention subgroup: g = 0.58 [0.34, 0.83],
    I^2 = 0%.

    CORRECTION: the previous version substituted an isochronic tone on a
    200 Hz carrier.  This preset now replicates the published stimulus:
    binaural, 340 Hz carrier, white-noise background.

    Delivery: binaural beat, L 340 Hz / R 380 Hz, white noise.
    Headphones required.  Studied exposure: 3 min before, then during task.""",
    ),
]

SUITE_MINDWAR = [
    preset(
        label    = "Delta  2 Hz  --  BWR Manual: 'exhausting' delta environment",
        mode     = "binaural", beat_hz=2.0, carrier_hz=400.0,
        evidence = "Limited",
        desc     = """\
    BWR Manual: in a delta (1-4 Hz) environment, complex or creative work is
    "exhausting and unproductive".  Closest research: 1.5/4 Hz beats during
    a vigilance task worsened performance and mood vs 16/24 Hz (Lane 1998).
    Headphones required.  Evidence: Limited.""",
        detail   = """\
    SOURCE CHECK: the BWR Manual attributes its band descriptions to Michael
    Aquino's MindWar.  The provided MindWar text (Valley & Aquino 1980, with
    Aquino's 2003 annotations) contains no band-by-band descriptions.  Its
    only related passage is footnote 21 (see the infrasonic presets).

    RESEARCH VIEW: delta is the dominant rhythm of deep sleep and does not
    support waking cognition.  Lane et al. (1998) found 1.5/4 Hz beats during
    a 30-min vigilance task gave fewer hits, more false alarms and more
    negative mood than 16/24 Hz beats.  Delta beats heard BEFORE a stressful
    event reduced anxiety (Wahbeh 2007b; Padmanabhan 2005).

    Delivery: binaural beat, 400 Hz carrier.  Headphones required.""",
    ),
    preset(
        label    = "Theta  5.5 Hz  --  BWR Manual: 'emotion / frustration'",
        mode     = "binaural", beat_hz=5.5, carrier_hz=400.0,
        evidence = "Claim contradicted",
        desc     = """\
    BWR Manual: 4-7 Hz "heightens emotions, imagination and frustration".
    Research mostly contradicts this: theta beats REDUCED anxiety (Isik
    2017; Garcia-Argibay 2018), though 30 min of 7 Hz beats raised one
    depression subscale (Wahbeh 2007a).  Headphones required.""",
        detail   = """\
    SOURCE CHECK: the "high emotion, violence and frustration" description is
    not in the provided MindWar text (Valley & Aquino 1980 / Aquino 2003).

    RESEARCH VIEW
      Isik et al. (2017): 10 min of theta beats reduced pre-operative anxiety.
      Garcia-Argibay et al. (2018): anxiety subgroup g = 0.69.
      Wahbeh et al. (2007a): one 30-min session of 7 Hz beats raised the POMS
        depression subscale and lowered verbal recall (Chaieb et al. 2015).
      Garcia-Argibay et al. (2017): 5 Hz beats impaired memory encoding.

    Delivery: binaural beat, 400 Hz carrier.  Headphones required.""",
    ),
    preset(
        label    = "Alpha  10 Hz  --  BWR Manual: 'relaxed, cooperative' state",
        mode     = "binaural", beat_hz=10.0, carrier_hz=400.0,
        evidence = "Mixed (EEG) / Limited (anxiety)",
        desc     = """\
    BWR Manual: alpha (8-12 Hz) supports "relaxed, cooperative conversation"
    and pattern recognition.  Research links alpha to relaxed wakefulness, and
    10 Hz beats lowered anxiety (Weiland 2011).  No study tested cooperation.
    Headphones required.""",
        detail   = """\
    SOURCE CHECK: the cooperative-discussion description is not in the
    provided MindWar text.

    RESEARCH VIEW: alpha is the dominant rhythm of relaxed wakefulness.  Kim
    et al. (2023) found raised alpha power after 5 min of a 10 Hz beat on a
    250 Hz carrier; several other studies did not (Ingendoh et al. 2023).
    Weiland et al. (2011): 10 Hz beats in nature sounds lowered anxiety.
    For the EEG-replicating version use General suite, Alpha 10 Hz.

    Delivery: binaural beat, 400 Hz carrier.  Headphones required.""",
    ),
    preset(
        label    = "Beta  17 Hz  --  BWR Manual: 'anxiety' claim (not supported)",
        mode     = "monaural", beat_hz=17.0, carrier_hz=400.0,
        evidence = "Claim unsupported",
        desc     = """\
    BWR Manual: 17 Hz brings "anxiety and visual disturbances", best format
    mono.  No provided study reports this, and 16 Hz beats improved mood
    (Lane 1998).  Played as a monaural beat (400 + 417 Hz) so the beat is
    audible on speakers.""",
        detail   = """\
    SOURCE CHECK: neither the 17 Hz warning nor the "13-22 Hz deliberate
    effort, logical thought" description is in the provided MindWar text.

    RESEARCH VIEW: no study links 17 Hz sound to anxiety.  Lane et al. (1998)
    found 16/24 Hz beats produced less negative mood than 1.5/4 Hz beats.  No
    beta-band EEG entrainment has been demonstrated (Ingendoh et al. 2023).

    FORMAT CORRECTION: the previous version played a raw 17 Hz sine, which is
    infrasonic and inaudible on ordinary headphones and speakers.  A monaural
    beat delivers an audible 17 Hz rhythm on speakers.

    Delivery: monaural beat, 400 + 417 Hz in both channels.""",
    ),
    preset(
        label    = "High-carrier AM  12.5 Hz  --  MindWar 'subliminal' claim (corrected)",
        mode     = "hf_am", beat_hz=12.5, carrier_hz=SUBLIMINAL_CARRIER,
        evidence = "None (mechanism implausible)",
        desc     = """\
    An 18,500 Hz tone amplitude-modulated at 12.5 Hz.  It contains no 12.5 Hz
    energy, only tones at 18,487.5-18,512.5 Hz.  Listeners who cannot hear
    18.5 kHz (most adults) receive nothing.  Those who can hear a faint,
    high-pitched flutter, which is audible, not subliminal.  Evidence: none.""",
        detail   = """\
    MINDWAR TEXT (Valley & Aquino 1980, footnote 21): "Infrasound vibration
    (up to 20 Hz) can subliminally influence brain activity", citing a
    popular book (Playfair & Hill, The Cycles of Heaven, 1978).  No study in
    the provided literature supports this.  Even clearly audible beats
    entrain the EEG inconsistently (Ingendoh et al. 2023).

    PHYSICS: amplitude modulation moves no energy down to 12.5 Hz.  The
    signal consists of three tones at 18,487.5, 18,500 and 18,512.5 Hz.  The
    ear can only follow the 12.5 Hz envelope if it can hear the carrier.  The
    upper limit of hearing falls with age, and many adults cannot hear
    above roughly 15-17 kHz.  For those listeners nothing arrives.  For those
    who can hear it, it is an audible high whine.  This mode was previously
    labelled "subliminal" and "covert"; neither description is accurate.

    Delivery: high-carrier AM.  Use 96 kHz if the interface supports it
    (44.1 kHz can represent 18.5 kHz, but many DACs roll off above 18 kHz).
    Only play it for people who have agreed to listen.""",
    ),
    preset(
        label    = "Infrasonic  12.5 Hz  --  Direct sub-20 Hz pressure wave",
        mode     = "infrasonic", beat_hz=12.5, carrier_hz=12.5,
        evidence = "None",
        desc     = """\
    A true 12.5 Hz pressure wave with no audible carrier.  Detectable only
    at very high levels (roughly 90-100 dB SPL), mostly as pressure or
    vibration.  Needs a subwoofer rated below 20 Hz; ordinary speakers
    reproduce almost nothing.  Keep the level low.  Evidence: none.""",
        detail   = """\
    MINDWAR TEXT (footnote 21): infrasound "up to 20 Hz" is claimed to align
    brain activity to delta, theta, alpha or beta.  The citation is a popular
    book, and no provided study tests the claim.

    PHYSICS: hearing thresholds rise steeply below 20 Hz.  At 12.5 Hz a tone
    must reach roughly 90-100 dB SPL to be noticed, and is felt as pressure
    rather than heard as pitch.  Most consumer speakers and headphones cannot
    move enough air at 12.5 Hz.  Driving them hard at full scale can damage
    small drivers.

    LIMITATION: the passage in MindWar is about ELF electromagnetic waves and
    infrasound.  An acoustic wave is not an electromagnetic field and does
    not penetrate walls the way the text implies.

    Delivery: direct infrasonic output.  Sub-20 Hz subwoofer required.""",
    ),
    preset(
        label    = "ELF  57.5 Hz  --  Project Sanguine range (acoustic stand-in)",
        mode     = "isochronic", beat_hz=57.5, carrier_hz=400.0,
        evidence = "Inconclusive",
        desc     = """\
    MindWar footnote 21 mentions ELF waves "up to 100 Hz" and the Navy's
    Project Sanguine.  ELF there means electromagnetic waves; this is an
    acoustic 57.5 Hz rhythm instead.  A 57.3 Hz binaural beat had no EEG
    effect (Lopez-Caballero 2017).  Speakers or headphones.""",
        detail   = """\
    MINDWAR TEXT (footnote 21): "ELF waves up to 100 Hz ... can also be
    produced artificially (such as for the Navy's Project Sanguine for
    submarine communication)", and are said to be linked to physiological
    disorders and emotional distortion.  The 45-70 Hz range and the Becker
    quotation cited by the previous version are not in the provided text.

    RESEARCH VIEW: Lopez-Caballero & Escera (2017) presented a 57.3 Hz binaural
    beat and found no EEG power change (Ingendoh et al. 2023).  Project
    Sanguine was an electromagnetic antenna system.  Sound at the same
    frequency is a different physical phenomenon.

    Delivery: isochronic tone, 400 Hz carrier (raised from 200 Hz so that each
    pulse holds several complete carrier cycles).  Speakers or headphones.""",
    ),
]

SUITE_ALZHEIMERS = [
    preset(
        label    = "MIT GENUS  40 Hz  --  Auditory click train (1 ms, 10 kHz pips)",
        mode     = "isochronic", beat_hz=40.0, carrier_hz=10000.0, pulse_ms=1.0,
        evidence = "Strong (mice) / Limited (humans)",
        ref_s    = 3600, ref_src="Tsai lab protocol (1 h/day)",
        desc     = """\
    Tsai lab (MIT Picower) auditory GENUS: 1 ms tone pips at 10 kHz repeated
    40 times per second (Martorell 2019), 1 hour a day.  In Alzheimer's mouse
    models this reduced amyloid and tau and improved memory; light and sound
    together worked best.  Human efficacy is not established.
    Speakers or headphones.  Evidence: Strong (mice) / Limited (humans).""",
        detail   = """\
    Sources: MIT Picower Institute, "40Hz rhythms fight Alzheimer's at the
    cellular and molecular level".  Iaccarino et al. (2016) Nature, 40 Hz
    LIGHT/optogenetics.  Martorell et al. (2019) Cell 177:256, 40 Hz SOUND.

    MOUSE FINDINGS: 1 h a day of 40 Hz light for a week reduced amyloid
    plaques and tau (2016).  Longer stimulation prevented synapse loss, cell
    death and brain atrophy, and 40 Hz sound alone reduced amyloid and tau
    and improved learning and memory.  Light and sound together were
    especially effective (2019).  Proposed mechanisms: improved network
    connectivity, VIP-driven vessel pulsation and AQP4 channels increasing
    glymphatic (CSF) clearance, and less inflammatory microglia.

    HUMANS: pilot studies found 40 Hz light + sound safe and suggested
    preserved brain volume and cognition (Picower).  Ismail et al. (2018):
    40 Hz LIGHT, 2 x 60 min a day for 10 days in 6 patients, produced no
    change in amyloid PET (p > .48); any effect was < 20%, far from the 50%
    seen in mice.  Cognito Therapeutics has launched a pivotal trial.

    STIMULUS: 1 ms tones at 10 kHz, one every 25 ms (Martorell et al. 2019).
    These parameters come from the original paper, not the Picower page.
    HEARING NOTE: many older adults hear 10 kHz poorly.  If the clicks sound
    faint, use the 400 Hz isochronic preset: 40 Hz responses are about 3x
    larger at a 250 Hz carrier than at 4 kHz (Ross, via Chaieb 2015).

    Delivery: 40 Hz pulse train.  Speakers or headphones, at a moderate,
    comfortable level.  Protocol: 1 hour daily.""",
    ),
    preset(
        label    = "40 Hz Monaural beat  --  440 + 480 Hz, speaker-compatible",
        mode     = "monaural", beat_hz=40.0, carrier_hz=440.0,
        evidence = "Limited",
        ref_s    = 300, ref_src="Engelbregt 2021",
        desc     = """\
    Both tones (440 + 480 Hz) in both ears, producing a physical 40 Hz beat
    that works on speakers.  In Engelbregt (2021) this was the only condition
    that raised 40 Hz EEG power (at Fz), but it also INCREASED Flanker
    errors.  Speakers or headphones.  Evidence: Limited.""",
        detail   = """\
    A monaural beat is two tones summed before reaching the ear: a sinusoidal
    100% amplitude modulation at the difference frequency.  It is demodulated
    in the cochlea, whereas binaural beats arise in the superior olivary
    complex (Chaieb et al. 2015).

    EVIDENCE
      Engelbregt et al. (2021): the MB condition (440 + 480 Hz, both ears)
        was the only one with higher 40 Hz power (Fz, vs pink noise and vs
        binaural).  It also produced MORE Flanker errors than pink noise
        (partial eta^2 = .47).
      Draganova et al. (2008): steady-state responses to monaural beats were
        about 5x larger than to binaural beats (Chaieb et al. 2015).
      Orozco Perez et al. (2020): at 40 Hz the binaural response was the
        larger of the two (Ingendoh et al. 2023).  The findings conflict.

    Delivery: monaural beat, 440 + 480 Hz in both channels.
    Speakers or headphones.  Studied exposure: 5 min (Engelbregt).""",
    ),
    preset(
        label    = "40 Hz Binaural  --  Headphones (440/480 Hz, Engelbregt 2021)",
        mode     = "binaural", beat_hz=40.0, carrier_hz=440.0,
        evidence = "Limited (for entrainment)",
        ref_s    = 300, ref_src="Engelbregt 2021",
        desc     = """\
    The Engelbregt (2021) binaural stimulus.  It improved Flanker accuracy but
    produced NO consistent 40/45 Hz EEG increase, so a benefit, if any, came
    without gamma entrainment, the mechanism GENUS relies on.  Prefer the
    click-train or monaural presets for entrainment.  Headphones required.""",
        detail   = """\
    Sources: Engelbregt et al. (2021); Schwarz & Taylor (2005); Grose & Mamo
    (2012), as summarised in Engelbregt et al. (2021).

    40 Hz BINAURAL RESPONSES: Schwarz & Taylor (2005) measured a 40 Hz
    steady-state response to a 380/420 Hz binaural beat.  Grose & Mamo (2012)
    found frequency-following responses at 390/430 Hz and, more weakly, at
    810/850 Hz.  Schwarz & Taylor found the 40 Hz binaural response was
    undetectable with carriers above 3 kHz (Chaieb et al. 2015).

    WITH THIS CARRIER PAIR: Engelbregt et al. (2021) found no consistent 40 or
    45 Hz power increase in frontal or temporal electrodes during 440/480 Hz
    binaural beats, despite fewer Flanker errors (partial eta^2 = .14).

    CORRECTION: the previous version stated that an ASSR was "confirmed at
    440/480 Hz" in Engelbregt (2021), and credited Schwarz & Taylor with the
    390/430 Hz pair.  Engelbregt found no such response, and the 390/430 Hz
    pair is from Grose & Mamo.

    Delivery: binaural beat.  Headphones required.  Studied exposure: 5 min.""",
    ),
    preset(
        label    = "40 Hz Isochronic  --  Low-carrier alternative (400 Hz tone)",
        mode     = "isochronic", beat_hz=40.0, carrier_hz=400.0,
        evidence = "Limited (extrapolated)",
        ref_s    = 3600, ref_src="GENUS daily dose (mice)",
        desc     = """\
    40 Hz on/off pulses of a 400 Hz tone.  A low-pitched alternative to the
    10 kHz click train for listeners with high-frequency hearing loss: 40 Hz
    responses are larger at low carriers (Ross, via Chaieb 2015).  Not the
    stimulus used in the mouse studies.  Speakers or headphones.""",
        detail   = """\
    Ross et al. (reviewed by Chaieb et al. 2015) found 40 Hz steady-state
    responses about three times larger with a 250 Hz carrier than with a
    4000 Hz carrier.  This preset trades the exact Tsai-lab stimulus (10 kHz
    pips) for one that elderly listeners can hear clearly.

    No study in the provided literature tested this exact stimulus for
    Alzheimer's outcomes.  The 1 h reference is the mouse GENUS dose.

    CORRECTION: this slot was previously "40 Hz Daily Maintenance (15 min)",
    justified by Ismail et al. (2018).  That study used light, 2 x 60 min a
    day, and found no effect in 10 days.  It gives no support for 15-minute
    sessions.

    Delivery: isochronic tone, 400 Hz carrier, 50% duty, 40 Hz.""",
    ),
    preset(
        label    = "40 Hz Parametric  --  Ultrasonic carriers (specialist hardware)",
        mode     = "parametric", beat_hz=40.0, carrier_hz=PARAMETRIC_BASE,
        evidence = "None (untested format)",
        desc     = """\
    Two ultrasonic tones (40,000 + 40,040 Hz).  Only a dedicated parametric
    speaker array, driven at very high ultrasonic levels, demodulates them
    in air into an audible 40 Hz tone.  Ordinary speakers and DACs produce
    nothing audible.  Needs a 96 kHz+ interface.  Evidence: none.""",
        detail   = """\
    PHYSICS: two ultrasonic tones produce an audible difference tone only
    through the nonlinearity of air at very high sound pressure, which is
    the principle of parametric speaker arrays.  The result is a 40 Hz tone
    (with harmonics), not an amplitude-modulated carrier.  At ordinary
    playback levels the demodulated component is negligible.

    HARDWARE: requires a 96 kHz or faster interface whose output filter
    passes 40 kHz, and an ultrasonic transducer array.  Keep exposure within
    occupational ultrasound guidelines.

    No clinical data exist for this delivery format.""",
    ),
    preset(
        label    = "Upper Gamma  60 Hz  --  Isochronic (no specific evidence)",
        mode     = "isochronic", beat_hz=60.0, carrier_hz=400.0,
        evidence = "Inconclusive",
        desc     = """\
    60 Hz lies in the upper gamma range.  The 40 Hz evidence does not
    extrapolate to it: a 57.3 Hz binaural beat produced no EEG change
    (Lopez-Caballero 2017), and the GENUS studies did not use 60 Hz.
    Speakers or headphones.  Evidence: Inconclusive.""",
        detail   = """\
    60 Hz sits in the upper gamma range associated with perceptual binding
    and local cortical processing.  The Tsai-lab work targets 40 Hz
    specifically.  Lopez-Caballero & Escera (2017) found no EEG power change
    for a 57.3 Hz binaural beat (Ingendoh et al. 2023).

    Note: mains hum is 60 Hz in the Americas, so this rhythm may sound like
    electrical hum.

    Delivery: isochronic tone, 400 Hz carrier (raised from 200 Hz so each
    8.3 ms pulse holds several complete carrier cycles).""",
    ),
]

SUITES = [
    ("General Brainwave Suite", SUITE_GENERAL),
    ("Attention & Cognition Suite", SUITE_ATTENTION),
    ("MindWar / BWR Manual Suite", SUITE_MINDWAR),
    ("Alzheimer's / 40 Hz Suite", SUITE_ALZHEIMERS),
]

# ---------------------------------------------------------------------------
#  SIGNAL GENERATION
#
#  Everything is computed from the absolute sample index, so blocks join
#  seamlessly and the output is identical for any block size.  The self-test
#  checks that property for every preset.
# ---------------------------------------------------------------------------

def duty_of(p):
    if p.get("pulse_ms"):
        return min(1.0, p["pulse_ms"] * 1e-3 * p["beat_hz"])
    return p["duty"]


def _sine(f, t):
    return np.sin(2 * np.pi * f * t)


def _pulse_ramp(rate, duty, ramp_s=0.005):
    """Raised-cosine edge length: 5 ms, but at most a quarter of the on-time."""
    return min(ramp_s, duty / rate / 4.0)


def pulse_gate(t, rate, duty):
    """Isochronic gate with raised-cosine rise AND fall, from absolute time."""
    if duty >= 1.0:
        return np.ones_like(t)
    period = 1.0 / rate
    on     = duty * period
    ramp   = _pulse_ramp(rate, duty)
    tau    = np.mod(t, period)
    g      = (tau < on).astype(np.float64)
    rise   = tau < ramp
    g[rise] = 0.5 - 0.5 * np.cos(np.pi * tau[rise] / ramp)
    fall   = (tau < on) & (tau > on - ramp)
    g[fall] = 0.5 - 0.5 * np.cos(np.pi * (on - tau[fall]) / ramp)
    return g


def tone_block(p, t, nch=2):
    """Unit-peak tone for one block.  Returns (n, nch) float64."""
    m, c, b = p["mode"], p["carrier_hz"], p["beat_hz"]
    if m == "binaural":
        return np.column_stack((_sine(c, t), _sine(c + b, t)))
    if m in ("monaural", "parametric"):
        x = 0.5 * (_sine(c, t) + _sine(c + b, t))
    elif m in ("isochronic", "surround"):
        x = _sine(c, t) * pulse_gate(t, b, duty_of(p))
    elif m == "hf_am":
        x = _sine(c, t) * (0.5 + 0.5 * np.cos(2 * np.pi * b * t))
    elif m == "infrasonic":
        x = _sine(b, t)
    else:                                   # mono
        x = _sine(c, t)
    return np.repeat(x[:, None], nch if m == "surround" else 2, axis=1)


# Pink-noise IIR filter (J. O. Smith, "Generation of pink noise"): -3 dB/octave
# within ~0.3 dB over the audio band.
_PINK_B = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
_PINK_A = [1.0, -2.494956002, 2.017265875, -0.522189400]


class NoiseSource:
    """Continuous white or pink noise at a fixed RMS level.  The filter state
    carries across blocks, so there are no seams and no per-block gain jumps."""

    def __init__(self, kind, seed=None):
        self.kind = kind
        self.rng  = np.random.default_rng(seed)
        self.gain = NOISE_RMS
        if kind == "pink":
            probe = sps.lfilter(_PINK_B, _PINK_A, np.random.default_rng(0).standard_normal(1 << 17))
            self.gain = NOISE_RMS / np.sqrt(np.mean(probe[8192:] ** 2))
            self.zi   = np.zeros(len(_PINK_A) - 1)

    def block(self, n):
        w = self.rng.standard_normal(n)
        if self.kind == "pink":
            w, self.zi = sps.lfilter(_PINK_B, _PINK_A, w, zi=self.zi)
        return w * self.gain


class Renderer:
    """Produces successive blocks of a session: tone, optional masking noise,
    fixed gain, fade-in, scheduled fade-out and an on-demand release fade."""

    def __init__(self, p, sr, amplitude=0.8, noise=None, noise_lvl=0.0,
                 total=None, nch=2, fade_s=FADE_S, seed=None):
        self.p, self.sr, self.amp, self.total = p, sr, amplitude, total
        self.nch   = nch if p["mode"] == "surround" else 2
        self.lvl   = noise_lvl if noise else 0.0
        self.noise = NoiseSource(noise, seed) if noise and noise_lvl > 0 else None
        self.fade  = int(fade_s * sr)
        self.rel_n = int(STOP_FADE_S * sr)
        self.rel_at = None
        self.pos   = 0

    def release(self):
        """Begin a short fade-out from the current position (used on Ctrl+C)."""
        if self.rel_at is None:
            self.rel_at = self.pos

    def next_block(self, n=BLOCK_SIZE):
        end = self.pos + n
        if self.total is not None:
            end = min(end, self.total)
        if self.rel_at is not None:
            end = min(end, self.rel_at + self.rel_n)
        n = end - self.pos
        if n <= 0:
            return None
        k = np.arange(self.pos, end, dtype=np.float64)
        x = tone_block(self.p, k / self.sr, self.nch)
        if self.noise is not None:
            x = x * (1.0 - self.lvl) + self.noise.block(n)[:, None] * self.lvl
        g = np.full(n, self.amp)
        if self.fade > 0:
            g *= np.minimum(1.0, k / self.fade)
            if self.total is not None:
                g *= np.minimum(1.0, (self.total - k) / self.fade)
        if self.rel_at is not None:
            g *= np.clip(1.0 - (k - self.rel_at) / self.rel_n, 0.0, 1.0)
        self.pos = end
        return np.clip(x * g[:, None], -1.0, 1.0).astype(np.float32)


def render(p, sr, seconds, block=BLOCK_SIZE, **kw):
    """Render a whole session into memory (used by the self-test)."""
    r = Renderer(p, sr, total=int(seconds * sr), **kw)
    out = []
    while (blk := r.next_block(block)) is not None:
        out.append(blk)
    return np.concatenate(out)

# ---------------------------------------------------------------------------
#  PARAMETER CHECKS
# ---------------------------------------------------------------------------

def signal_top_hz(p):
    m, c, b = p["mode"], p["carrier_hz"], p["beat_hz"]
    if m == "infrasonic":
        return b
    if m == "mono":
        return c
    return c + b


def check_params(p, sr):
    """Return (errors, warnings) for a preset at a given sample rate."""
    errors, warns = [], []
    m, c, b = p["mode"], p["carrier_hz"], p["beat_hz"]
    nyq = sr / 2
    top = signal_top_hz(p)
    if top >= nyq:
        errors.append(f"{top:g} Hz is at or above the Nyquist limit ({nyq:g} Hz) "
                      f"for {sr} Hz audio; it would alias.")
    elif top > 0.9 * nyq:
        warns.append(f"{top:g} Hz is close to the Nyquist limit; many DACs filter it out.")
    if m in ("isochronic", "surround"):
        d = duty_of(p)
        if not 0.0 < d <= 1.0:
            errors.append("Duty cycle must be greater than 0 and at most 1.")
        elif c * d / b < 2:
            warns.append(f"Each pulse holds only {c*d/b:.1f} carrier cycles.  A carrier of "
                         f"at least {math.ceil(2*b/d)} Hz gives a clean tone.")
    if m == "binaural":
        if b > BINAURAL_LIMIT:
            warns.append(f"{b:g} Hz is above the ~{BINAURAL_LIMIT:g} Hz limit for hearing a "
                         "binaural beat (Perrott & Nelson 1969).  Brain responses to 40 Hz "
                         "beats are still measurable (Schwarz & Taylor 2005).")
        if c + b > BINAURAL_CARRIER_MAX:
            warns.append("Binaural beats need tones below ~1000 Hz (Licklider 1950; Oster "
                         "1973); 400 Hz is the usual recommendation (Goodin 2012).")
    if m in ("mono", "infrasonic"):
        f = c if m == "mono" else b
        if f < 20:
            warns.append(f"{f:g} Hz is infrasonic.  Most headphones and speakers cannot "
                         "reproduce it audibly, and it is detected only at very high levels.  "
                         "Keep the volume low to protect small drivers.  For an audible "
                         f"{f:g} Hz rhythm use the monaural or isochronic mode.")
    if m == "hf_am" and c > 15000:
        warns.append("Many adults cannot hear above ~15-17 kHz.  A listener who cannot hear "
                     f"the carrier receives no {b:g} Hz modulation at all.")
    if m == "parametric":
        warns.append("Needs a parametric (ultrasonic) speaker array; ordinary speakers "
                     "produce nothing audible.")
    return errors, warns

# ---------------------------------------------------------------------------
#  FREQUENCY ANALYSIS
# ---------------------------------------------------------------------------

def _spectrum(x, sr):
    """Amplitude spectrum of a Hann-windowed, DC-removed signal."""
    x = x - np.mean(x)
    w = np.hanning(len(x))
    X = np.abs(np.fft.rfft(x * w)) * 2.0 / w.sum()
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    return f, X


def _interp(X, i):
    """Parabolic interpolation of a spectral peak on log magnitude (bins)."""
    if 0 < i < len(X) - 1:
        a, b, c = np.log(X[i - 1:i + 2] + 1e-30)
        den = a - 2 * b + c
        if den != 0:
            return 0.5 * (a - c) / den
    return 0.0


def _tonal_peaks(f, X, fmin, fmax, k=3, floor_db=-40.0):
    df = f[1] - f[0]
    lo = max(1, int(np.searchsorted(f, fmin)))
    hi = int(np.searchsorted(f, fmax))
    if hi - lo < 3:
        return []
    idx, _ = sps.find_peaks(X[lo:hi], distance=max(1, int(round(0.5 / df))))
    if len(idx) == 0:
        return []
    idx = idx + lo
    idx = idx[np.argsort(X[idx])[::-1]][:k]
    ref = X[idx[0]]
    peaks = []
    for i in idx:
        rel = 20 * np.log10(X[i] / ref)
        if rel >= floor_db:
            peaks.append((f[i] + _interp(X, i) * df, rel))
    return peaks


def _envelope_modulation(x, sr, fmin=0.5, fmax=120.0):
    """Lowest prominent amplitude-modulation rate and its depth, or None."""
    n = len(x)
    env = np.abs(sps.hilbert(x - np.mean(x), spfft.next_fast_len(n))[:n])
    q = max(1, int(sr // 1000))                 # decimate envelope to ~1 kHz
    m = n // q * q
    if m < 4 * q:
        return None
    env = env[:m].reshape(-1, q).mean(axis=1)
    esr = sr / q
    mean = env.mean()
    if mean <= 1e-9:
        return None
    f, E = _spectrum(env, esr)
    band = np.where((f >= fmin) & (f <= min(fmax, esr / 2)))[0]
    if len(band) < 8:
        return None
    floor = np.median(E[band])
    idx, _ = sps.find_peaks(E[band], height=max(25 * floor, 0.01 * mean))
    if len(idx) == 0:
        return None
    idx = band[idx]
    strongest = E[idx].max()
    i = min(j for j in idx if E[j] >= 0.3 * strongest)   # fundamental, not a harmonic
    rate = f[i] + _interp(E, i) * (f[1] - f[0])
    # Depth = how far the cycle-averaged envelope dips below its peak:
    # 100% for on/off pulses and monaural beats, matching embed mode's depth.
    phase = np.floor(np.mod(np.arange(len(env)) * rate / esr, 1.0) * 64).astype(int)
    cycle = np.bincount(phase, env, 64) / np.maximum(np.bincount(phase, None, 64), 1)
    return rate, 1.0 - cycle.min() / cycle.max()


def analyse(data, sr, max_ch=8):
    """Analyse a (frames, channels) float array.  Returns a dict."""
    res = dict(sr=sr, seconds=len(data) / sr, df=sr / len(data),
               nch=data.shape[1], channels=[], binaural=None)
    for ch in range(min(data.shape[1], max_ch)):
        x = data[:, ch].astype(np.float64)
        rms = float(np.sqrt(np.mean((x - x.mean()) ** 2)))
        if rms < 1e-6:
            res["channels"].append(dict(silent=True))
            continue
        f, X = _spectrum(x, sr)
        res["channels"].append(dict(
            silent=False, rms_db=20 * np.log10(rms),
            peaks=_tonal_peaks(f, X, 0.5, 0.999 * sr / 2),
            mod=_envelope_modulation(x, sr)))
    chs = res["channels"]
    if len(chs) >= 2 and not chs[0]["silent"] and not chs[1]["silent"] \
            and chs[0]["peaks"] and chs[1]["peaks"]:
        fl, fr = chs[0]["peaks"][0][0], chs[1]["peaks"][0][0]
        shared = any(abs(fl - g) < 0.05 for g, _ in chs[1]["peaks"]) or \
                 any(abs(fr - g) < 0.05 for g, _ in chs[0]["peaks"])
        if not shared and max(fl, fr) < 1500 and 0.05 < abs(fr - fl) <= 45:
            res["binaural"] = (fl, fr, abs(fr - fl))
    return res


def detect_frequency(file_path, duration=60):
    """Analyse up to `duration` seconds of an audio file and print a report.
    Returns the analysis dict (None on error)."""
    if not os.path.isfile(file_path):
        print(f"  Error: file not found: {file_path}")
        return None
    try:
        info = sf.info(file_path)
        frames = min(info.frames, int(info.samplerate * duration))
        if frames < info.samplerate // 10:
            print("  Error: file is too short to analyse (< 0.1 s).")
            return None
        data, sr = sf.read(file_path, frames=frames, always_2d=True, dtype="float64")
    except Exception as e:
        print(f"  Error reading audio: {e}")
        return None

    res = analyse(data, sr)
    names = {0: "left", 1: "right"} if data.shape[1] == 2 else {}
    print(f"  Sample rate : {sr} Hz, {data.shape[1]} channel(s)")
    print(f"  Analysed    : {res['seconds']:.1f} s  (resolution {res['df']:.3f} Hz)")
    for i, ch in enumerate(res["channels"]):
        print()
        tag = f" ({names[i]})" if i in names else ""
        if ch["silent"]:
            print(f"  Channel {i+1}{tag}: silent")
            continue
        print(f"  Channel {i+1}{tag}   level {ch['rms_db']:.1f} dBFS RMS")
        peaks = ", ".join(f"{fq:.2f} Hz ({rel:+.1f} dB)" for fq, rel in ch["peaks"])
        print(f"    Strongest tones : {peaks or 'none'}")
        if ch["mod"]:
            print(f"    Amplitude mod.  : {ch['mod'][0]:.2f} Hz  (level dips {ch['mod'][1]*100:.0f}%)")
        else:
            print("    Amplitude mod.  : none detected (0.5-120 Hz)")
    if res["nch"] > len(res["channels"]):
        print(f"\n  ({res['nch'] - len(res['channels'])} further channels not analysed)")
    print()
    if res["binaural"]:
        fl, fr, d = res["binaural"]
        print(f"  Binaural offset : {d:.2f} Hz  (left {fl:.2f} Hz / right {fr:.2f} Hz)")
        if d > BINAURAL_LIMIT:
            print(f"    Above the ~{BINAURAL_LIMIT:g} Hz binaural percept limit (Perrott & Nelson 1969).")
    mods = [c["mod"][0] for c in res["channels"] if not c["silent"] and c["mod"]]
    if res["binaural"] and not mods:
        print("  Looks like      : binaural beat (pure tone per ear, no physical beat)")
    elif mods:
        print(f"  Looks like      : monaural / isochronic / AM rhythm at {mods[0]:.2f} Hz")
    first = next((c for c in res["channels"] if not c["silent"] and c["peaks"]), None)
    if first:
        print(f"  Dominant freq.  : {first['peaks'][0][0]:.2f} Hz")
    return res

# ---------------------------------------------------------------------------
#  PLAYBACK AND EXPORT
# ---------------------------------------------------------------------------

stop_event = threading.Event()


def _sigint(sig, frame):
    if stop_event.is_set():
        raise KeyboardInterrupt
    stop_event.set()
    sys.stdout.write("\n  Stopping (fading out) -- Ctrl+C again to stop immediately.\n")
    sys.stdout.flush()


def fmt(s):
    s = int(max(0, s)); h, r = divmod(s, 3600); m, sc = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{sc:02d}"


def _status_line(elapsed, total_s, p):
    goal = total_s or p.get("ref_s") or 0
    pct  = min(elapsed / goal * 100, 100) if goal else 0
    bar  = "=" * int(pct / 5) + "-" * (20 - int(pct / 5))
    rem  = f"  rem {fmt(total_s - elapsed)}" if total_s else "  continuous"
    return f"\r  {fmt(elapsed)}  [{bar}] {pct:5.1f}%{rem}   "


def _monitor(p, total_s, pretask, done, t0):
    """Progress line, reference-exposure notice and pre-task alert."""
    ref, ref_hit, alert_hit = p.get("ref_s"), False, False
    while not done.wait(1.0):
        e = time.time() - t0
        sys.stdout.write(_status_line(e, total_s, p))
        if ref and not ref_hit and e >= ref:
            ref_hit = True
            sys.stdout.write(f"\n  *** {fmt(ref)} reached: the exposure used in {p['ref_src']}.\n")
        if pretask and not alert_hit and e >= pretask:
            alert_hit = True
            sys.stdout.write("\n\n  ╔══════════════════════════════════╗\n"
                             "  ║  >>> BEGIN YOUR TASK NOW <<<     ║\n"
                             "  ║  Pre-task induction complete     ║\n"
                             "  ╚══════════════════════════════════╝\n\n")
        sys.stdout.flush()
    sys.stdout.write("\n")


def _surround_channels(sd):
    try:
        d = sd.query_devices(sd.default.device[1], "output")
        return max(2, int(d["max_output_channels"]))
    except Exception:
        return 2


def play_session(p, cfg, duration, sr, pretask=0.0):
    """Play through the default output device.  Returns elapsed seconds, or
    None if playback could not start."""
    sd = _load_sounddevice()
    if sd is None:
        return None
    nch = _surround_channels(sd) if p["mode"] == "surround" else 2
    try:
        sd.check_output_settings(samplerate=sr, channels=nch, dtype=DTYPE)
    except Exception as e:
        print(f"  The output device cannot play {sr} Hz / {nch} channels: {e}")
        print("  Choose another sample rate, or export to a file instead.")
        return None

    total = int(duration * sr) if duration else None
    r = Renderer(p, sr, cfg["amplitude"], cfg["noise"], cfg["noise_lvl"], total, nch)
    q = queue.Queue(maxsize=PREFILL * 2)
    finished, errors = threading.Event(), []
    underruns = [0]

    def put(item):
        while not finished.is_set():
            try:
                q.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def producer():
        try:
            while True:
                if stop_event.is_set():
                    r.release()
                blk = r.next_block()
                if blk is None:
                    put(None)
                    return
                if len(blk) < BLOCK_SIZE:
                    blk = np.vstack((blk, np.zeros((BLOCK_SIZE - len(blk), blk.shape[1]), np.float32)))
                if not put(blk):
                    return
        except Exception as e:
            errors.append(e)
            put(None)

    def callback(out, frames, ti, status):
        if status.output_underflow:
            underruns[0] += 1
        try:
            blk = q.get_nowait()
        except queue.Empty:
            out.fill(0)
            underruns[0] += 1
            return
        if blk is None:
            out.fill(0)
            raise sd.CallbackStop()
        out[:] = blk

    stop_event.clear()
    threading.Thread(target=producer, daemon=True).start()
    deadline = time.time() + 5
    while q.qsize() < PREFILL and not errors and time.time() < deadline:
        time.sleep(0.005)
    if errors:
        raise errors[0]

    old = signal.signal(signal.SIGINT, _sigint)
    t0 = time.time()
    mon = threading.Thread(target=_monitor, args=(p, duration, pretask, finished, t0), daemon=True)
    try:
        with sd.OutputStream(samplerate=sr, channels=nch, blocksize=BLOCK_SIZE,
                             dtype=DTYPE, callback=callback,
                             finished_callback=finished.set):
            mon.start()
            stop_seen = None
            while not finished.wait(0.2):
                if stop_event.is_set():
                    stop_seen = stop_seen or time.time()
                    if time.time() - stop_seen > STOP_FADE_S + 3:
                        break       # safety net; leaving the block aborts the stream
    except KeyboardInterrupt:
        print("\n  Stopped immediately.")
    finally:
        finished.set()
        signal.signal(signal.SIGINT, old)
        mon.join(timeout=2)
    if errors:
        raise errors[0]
    if underruns[0]:
        print(f"  Note: {underruns[0]} audio buffer underrun(s) -- the system could not keep up.")
    return time.time() - t0


EXPORT_FORMATS = {".wav": "PCM_24", ".flac": "PCM_24", ".ogg": None, ".aiff": "PCM_24"}


def export_session(p, cfg, duration, sr, path, nch=2):
    """Render a timed session to an audio file in blocks."""
    ext = os.path.splitext(path)[1].lower()
    total = int(duration * sr)
    r = Renderer(p, sr, cfg["amplitude"], cfg["noise"], cfg["noise_lvl"], total, nch)
    with sf.SoundFile(path, "w", samplerate=sr, channels=r.nch, subtype=EXPORT_FORMATS.get(ext)) as f:
        last = -1
        while (blk := r.next_block(1 << 16)) is not None:
            f.write(blk)
            pct = int(r.pos * 100 / total)
            if pct != last:
                sys.stdout.write(f"\r  Rendering... {pct:3d}%")
                sys.stdout.flush()
                last = pct
    print(f"\n  Written: {path}  ({fmt(duration)}, {sr} Hz, {r.nch} ch)")

# ---------------------------------------------------------------------------
#  EMBED MODE
# ---------------------------------------------------------------------------

def embed_file(inp, out, beat, depth):
    """Apply amplitude modulation at `beat` Hz to an audio file.  Gain swings
    between 1 - depth and 1, so the result never clips where the input did not.
    Keeps the input's sample format when the output container allows it."""
    with sf.SoundFile(inp) as fin:
        sr = fin.samplerate
        fmt_out = os.path.splitext(out)[1].lstrip(".").upper() or fin.format
        subtype = fin.subtype if sf.check_format(fmt_out, fin.subtype) else None
        with sf.SoundFile(out, "w", samplerate=sr, channels=fin.channels,
                          format=fmt_out, subtype=subtype) as fout:
            pos = 0
            for blk in fin.blocks(blocksize=1 << 16, always_2d=True, dtype="float64"):
                t = (np.arange(len(blk)) + pos) / sr
                g = 1.0 - depth * 0.5 * (1.0 - np.cos(2 * np.pi * beat * t))
                fout.write(blk * g[:, None])
                pos += len(blk)
    return sr, pos


def run_embed():
    print()
    rule("=")
    print("  Embed Mode -- amplitude-modulate an existing recording")
    rule("=")
    print()
    print("  Applies a periodic volume change at the chosen rate to an audio")
    print("  file (WAV, FLAC, OGG, AIFF).  At the default 15% depth it is heard")
    print("  as a gentle tremolo on sustained sounds.  It is not inaudible.")
    print()
    print("  Evidence: no provided study tested modulated music.  Garcia-Argibay")
    print("  (2018) found beats embedded in music were LESS effective than")
    print("  unmasked beats (b = 0.40, p = .022).")
    print()
    inp = input("  Input file path  : ").strip().strip('"')
    if not os.path.isfile(inp):
        print("  Error: file not found."); return
    stem, ext = os.path.splitext(inp)
    beat = ask_float("  Modulation rate (Hz) : ")
    dflt = f"{stem}_am{beat:g}Hz{ext}"
    out = input(f"  Output file path [{dflt}] : ").strip().strip('"') or dflt
    if os.path.abspath(out) == os.path.abspath(inp):
        print("  Error: output must differ from input."); return
    val = input("  Modulation depth 0.0-1.0  [0.15 = subtle / 1.0 = full] : ").strip()
    try:
        depth = max(0.0, min(1.0, float(val) if val else 0.15))
    except ValueError:
        depth = 0.15
    try:
        sr, n = embed_file(inp, out, beat, depth)
    except Exception as e:
        print(f"  Error processing audio: {e}"); return
    print()
    print(f"  Written to      : {out}")
    print(f"  Sample rate     : {sr} Hz")
    print(f"  Duration        : {n/sr:.1f} s")
    print(f"  Modulation rate : {beat:g} Hz")
    print(f"  Modulation depth: {depth*100:.0f}%  (gain {1-depth:.2f}-1.00, no clipping added)")

# ---------------------------------------------------------------------------
#  DETECT MODE
# ---------------------------------------------------------------------------

def run_detect():
    print()
    rule("=")
    print("  Analyse Audio -- tones, binaural offset and modulation rate")
    rule("=")
    print()
    print("  Reports the strongest tones in each channel, the left/right")
    print("  frequency offset (binaural beat) and any amplitude-modulation")
    print("  rhythm (monaural beats, isochronic pulses).  Use it to check")
    print("  what a downloaded 'brainwave' track actually contains.")
    print()
    inp = input("  Audio file path : ").strip().strip('"')
    if not os.path.isfile(inp):
        print("  Error: file not found."); return
    v = input("  Seconds to analyse [60]: ").strip()
    try:
        dur = float(v) if v else 60.0
    except ValueError:
        dur = 60.0
    print()
    detect_frequency(inp, dur)

# ---------------------------------------------------------------------------
#  CONSOLE UI
# ---------------------------------------------------------------------------

W = 64


def rule(c=""):
    print(c * W)


def print_wrapped(tag, text):
    print(textwrap.fill(text, width=W + 8, initial_indent="  " + tag,
                        subsequent_indent="  " + " " * len(tag)))


def ask_int(prompt, lo, hi, default=None):
    while True:
        v = input(prompt).strip()
        if not v and default is not None:
            return default
        try:
            v = int(v)
            if lo <= v <= hi:
                return v
        except ValueError:
            pass


def ask_float(prompt, default=None, lo=0.0):
    while True:
        v = input(prompt).strip()
        if not v and default is not None:
            return default
        try:
            v = float(v)
            if v > lo:
                return v
        except ValueError:
            pass


def ask_hms(prompt, default_s=None):
    """Accept minutes ('17', '2.5') or [HH:]MM:SS.  Returns seconds."""
    hint = f" [{fmt(default_s)}]" if default_s else ""
    while True:
        v = input(f"{prompt} (minutes or HH:MM:SS){hint}: ").strip()
        if not v and default_s:
            return float(default_s)
        try:
            if ":" in v:
                parts = [float(x) for x in v.split(":")]
                if len(parts) > 3 or any(x < 0 for x in parts):
                    raise ValueError
                s = 0.0
                for x in parts:
                    s = s * 60 + x
            else:
                s = float(v) * 60
            if s >= 0:
                return s
        except ValueError:
            pass


def choose_amplitude():
    print()
    rule("-")
    print("  Volume / amplitude  (0.1 = very quiet, 0.8 = default, 1.0 = full)")
    rule("-")
    print()
    print("  Set your system volume so the tone is comfortable and clearly")
    print("  audible, not loud (Engelbregt 2021: 'a comfortable level').")
    print("  Prolonged loud listening damages hearing.")
    v = input("\n  Amplitude [0.8]: ").strip()
    try:
        amp = max(0.05, min(1.0, float(v) if v else 0.8))
    except ValueError:
        amp = 0.8
    print(f"  Amplitude set to {amp:.2f}")
    return amp


def choose_noise(p):
    print()
    rule("-")
    print("  Masking noise (optional)")
    rule("-")
    print()
    used = {"pink": "pink noise", "white": "white noise"}.get(p["noise"])
    if p.get("ref_src"):
        print(f"  The cited study ({p['ref_src']}) used: {used or 'no masking noise'}.")
    print("  Garcia-Argibay (2018): noise-masked beats were not significantly")
    print("  less effective than unmasked ones (music masking was).")
    print("  Ingendoh (2023): no EEG study that embedded beats in pink noise")
    print("  found entrainment.")
    print()
    opts = [(None, 0.0), ("pink", 0.10), ("pink", 0.25), ("white", 0.10), ("white", 0.25)]
    print("  0  None")
    print("  1  Pink noise, subtle (10%)")
    print("  2  Pink noise, moderate (25%)")
    print("  3  White noise, subtle (10%)")
    print("  4  White noise, moderate (25%)")
    print("  5  Custom")
    dflt = 0
    if p["noise"]:
        dflt = opts.index((p["noise"], p["noise_lvl"])) if (p["noise"], p["noise_lvl"]) in opts else 2
    c = ask_int(f"\n  Enter number (0-5) [{dflt}]: ", 0, 5, dflt)
    if c < 5:
        kind, lvl = opts[c]
    else:
        kind = "white" if input("  Type (pink/white) [pink]: ").strip().lower().startswith("w") else "pink"
        try:
            lvl = max(0.0, min(1.0, float(input("  Level 0.0-1.0: ").strip())))
        except ValueError:
            lvl = DEFAULT_NOISE_LVL
    if kind and lvl > 0:
        print(f"  {kind.capitalize()} noise blend: {lvl*100:.0f}%")
        return kind, lvl
    print("  No masking noise.")
    return None, 0.0


def choose_output():
    print()
    rule("-")
    print("  Output")
    rule("-")
    print()
    print("  1  Play now through the default audio device")
    print("  2  Export to an audio file (WAV / FLAC / OGG)")
    return ask_int("\n  Enter number (1-2) [1]: ", 1, 2, 1)


def choose_duration(p, export=False):
    ref, timing = p.get("ref_s"), p.get("timing", "during")
    print()
    rule("-")
    print("  Session duration")
    rule("-")
    print()
    if ref:
        print(f"  Exposure used in the cited study: {fmt(ref)}  ({p['ref_src']})")
        print("  Garcia-Argibay (2018): longer exposure gave larger effects.")
        print()
    opts = [] if export else [("Continuous  (Ctrl+C to stop)", "cont")]
    opts.append(("Timed session", "timed"))
    if timing == "before_and_during":
        opts.append(("Pre-task induction, then keep playing during the task", "both"))
    elif timing == "before":
        opts.append(("Pre-task induction: play, then alert and stop", "before"))
    for i, (txt, _) in enumerate(opts, 1):
        print(f"  {i}  {txt}")
    if timing in ("before", "before_and_during"):
        print("     Garcia-Argibay (2018): exposure before, or before and during,")
        print("     the task gave larger effects than exposure during it only.")
    kind = opts[ask_int(f"\n  Enter number (1-{len(opts)}): ", 1, len(opts)) - 1][1]

    print()
    if kind == "cont":
        return 0.0, 0.0
    if kind == "before":
        s = ask_hms("  Induction length", ref)
        return (s, s) if s > 0 else (0.0, 0.0)
    pretask = 0.0
    if kind == "both":
        pretask = ask_hms("  Induction before the task", p.get("pre_s") or ref)
    total = ask_hms("  Total session length (induction + task)",
                    max(ref or 0, pretask + 600) if kind == "both" else ref)
    if total <= 0:
        if export:
            print("  Export needs a length; using 10 minutes.")
            return 600.0, 0.0
        print("  Zero entered.  Switching to continuous.")
        return 0.0, pretask
    if pretask >= total:
        print("  Induction is not shorter than the session; the alert will not fire.")
    if ref and total < ref:
        print(f"  Note: shorter than the {fmt(ref)} used in {p['ref_src']}.")
    return float(total), pretask


def choose_sr(p):
    m = p["mode"]
    if m == "parametric":
        print()
        rule("-")
        print("  Sample rate  (parametric mode)")
        rule("-")
        print()
        print("  The carriers are at 40,000 and 40,040 Hz, which needs at least")
        print("  96 kHz.  Your interface's output filter must also pass 40 kHz.")
        print()
        print("  1  96000 Hz   (recommended)")
        print("  2  192000 Hz")
        return {1: 96000, 2: 192000}[ask_int("\n  Enter number (1-2) [1]: ", 1, 2, 1)]
    if m == "hf_am":
        print()
        rule("-")
        print("  Sample rate  (high-carrier AM)")
        rule("-")
        print()
        print("  44.1 kHz can represent the carrier without aliasing, but many")
        print("  DAC output filters attenuate content above ~18 kHz.  96 kHz moves")
        print("  the carrier well inside the passband if the interface supports it.")
        print()
        print("  1  44100 Hz")
        print("  2  96000 Hz   (recommended)")
        return {1: 44100, 2: 96000}[ask_int("\n  Enter number (1-2) [2]: ", 1, 2, 2)]
    return SR_DEFAULT


def choose_suite():
    print()
    rule("=")
    print("  Select a frequency suite")
    rule("=")
    print()
    print("  1  General Brainwave Suite")
    print("       Delta, theta, Schumann, alpha, beta and gamma presets, each")
    print("       matched to the stimulus used in its cited study.")
    print()
    print("  2  Attention & Cognition Suite")
    print("       Study replications: Engelbregt 2021 | Malandrone 2022 |")
    print("       Kraus & Porubanova 2015 | Garcia-Argibay 2017 | Colzato 2017")
    print()
    print("  3  MindWar / BWR Manual Suite")
    print("       Frequencies from the BWR Manual's MindWar-derived framework,")
    print("       each checked against the research literature.")
    print()
    print("  4  Alzheimer's / 40 Hz Suite")
    print("       MIT GENUS click train, 40 Hz monaural / binaural / isochronic.")
    print()
    print("  5  Manual Configuration")
    print("       Specify mode, frequencies and duration directly.")
    print()
    print("  6  Embed Mode")
    print("       Amplitude-modulate an existing recording.")
    print()
    print("  7  Analyse Audio")
    print("       Measure tones, binaural offset and modulation in a file.")
    print()
    return ask_int("  Enter number (1-7): ", 1, 7)


def choose_preset(suite, title):
    print()
    rule("=")
    print(f"  {title}")
    rule("=")
    for i, p in enumerate(suite, 1):
        print()
        print(f"  {i}  {p['label']}")
        for line in p["desc"].splitlines():
            print(f"  {line}")
    print()
    rule("-")
    return suite[ask_int(f"  Enter number (1-{len(suite)}): ", 1, len(suite)) - 1]


def manual_config():
    print()
    rule("=")
    print("  Manual Configuration")
    rule("=")
    print()
    names = list(MODES)
    for i, m in enumerate(names, 1):
        print(f"  {i}  {m:<11} -- {MODES[m]}")
    print()
    mode = names[ask_int(f"  Enter number (1-{len(names)}): ", 1, len(names)) - 1]

    p = preset(mode=mode, carrier_hz=0.0, beat_hz=0.0, evidence="N/A")
    if mode == "mono":
        p["carrier_hz"] = ask_float("  Frequency (Hz): ")
        p["label"] = f"Manual mono {p['carrier_hz']:g} Hz"
    elif mode == "infrasonic":
        p["beat_hz"] = p["carrier_hz"] = ask_float("  Frequency (Hz, below 20): ")
        p["label"] = f"Manual infrasonic {p['beat_hz']:g} Hz"
    else:
        p["beat_hz"] = ask_float("  Beat / modulation frequency (Hz): ")
        default_c = {"hf_am": SUBLIMINAL_CARRIER, "parametric": PARAMETRIC_BASE}.get(mode, CARRIER_OPTIMAL)
        if mode == "parametric":
            p["carrier_hz"] = default_c
        else:
            label = "Lower tone (Hz)" if mode in ("binaural", "monaural") else "Carrier (Hz)"
            p["carrier_hz"] = ask_float(f"  {label} [{default_c:g}]: ", default_c)
        if mode in ("isochronic", "surround"):
            d = input("  Duty cycle 0.0-1.0 [0.5]: ").strip()
            try:
                p["duty"] = float(d) if d else 0.5
            except ValueError:
                p["duty"] = 0.5
        p["label"] = f"Manual {mode} {p['beat_hz']:g} Hz"

    sr = choose_sr(p)
    errors, warns = check_params(p, sr)
    for w in warns:
        print()
        print_wrapped("Warning: ", w)
    if errors:
        for e in errors:
            print()
            print_wrapped("Error: ", e)
        print("\n  Please re-enter the configuration.")
        return manual_config()
    p["sr"] = sr
    return p


def print_session_header(p, cfg, duration, sr, pretask):
    m, c, b = p["mode"], p["carrier_hz"], p["beat_hz"]
    print()
    rule("=")
    print(f"  {p['label']}")
    rule("=")
    if p.get("detail"):
        print()
        for line in p["detail"].splitlines():
            print(f"  {line}")
    print()
    rule("-")
    print(f"  Mode         : {m}")
    if m == "binaural":
        print(f"  Tones        : left {c:g} Hz / right {c+b:g} Hz  ->  {b:g} Hz beat")
    elif m in ("monaural", "parametric"):
        print(f"  Tones        : {c:g} + {c+b:g} Hz in both channels  ->  {b:g} Hz beat")
    elif m in ("isochronic", "surround"):
        d = duty_of(p)
        print(f"  Carrier      : {c:g} Hz, pulsed at {b:g} Hz")
        print(f"  Pulse        : {d/b*1000:.2f} ms on ({d*100:.0f}% duty), "
              f"{_pulse_ramp(b, d)*1000:.2f} ms raised-cosine edges")
    elif m == "hf_am":
        print(f"  Carrier      : {c:g} Hz, amplitude-modulated at {b:g} Hz")
    elif m == "infrasonic":
        print(f"  Frequency    : {b:g} Hz")
    else:
        print(f"  Frequency    : {c:g} Hz")
    print(f"  Sample rate  : {sr} Hz")
    print(f"  Fade         : {FADE_S:g} s in / {FADE_S:g} s out ({STOP_FADE_S:g} s on Ctrl+C)")
    print(f"  Volume       : {cfg['amplitude']:.2f}")
    if cfg["noise"]:
        print(f"  Noise        : {cfg['noise']} noise, {cfg['noise_lvl']*100:.0f}% blend")
    if pretask > 0:
        print(f"  Pre-task     : {fmt(pretask)} induction, then task-begin alert")
    print(f"  Duration     : {fmt(duration) if duration else 'Continuous  (Ctrl+C to stop)'}")
    if p.get("ref_s"):
        print(f"  Studied dose : {fmt(p['ref_s'])}  ({p['ref_src']})")
    if p.get("evidence"):
        print(f"  Evidence     : {p['evidence']}")
    timing_labels = {
        "during":            "during the task / activity",
        "before":            "before the task (induction)",
        "before_and_during": "before and during the task",
    }
    print(f"  Timing       : {timing_labels.get(p['timing'], p['timing'])} (as in the cited study)")
    rule("-")
    print()
    if m == "binaural":
        print("  Headphones required: each ear must hear only its own tone.")
    elif m == "infrasonic":
        print("  Needs a subwoofer that reaches below 20 Hz.  Keep the level low.")
    elif m == "hf_am":
        print("  Most adults hear little or nothing; younger listeners hear a")
        print("  high-pitched flutter.  Only play it for people who agreed to listen.")
    elif m == "parametric":
        print("  Needs a 96 kHz+ interface and a parametric (ultrasonic) speaker array.")
    elif m == "surround":
        print("  Plays on every output channel.  Everyone in the room will hear it,")
        print("  so only use it with people who have agreed to listen.")
    else:
        print("  Speakers or headphones.  Passive listening in a quiet room unless")
        print("  the study played it during a task.")


def run_tone_session(p):
    cfg = dict(amplitude=choose_amplitude())
    cfg["noise"], cfg["noise_lvl"] = choose_noise(p)
    out = choose_output()
    duration, pretask = choose_duration(p, export=(out == 2))
    sr = p.get("sr") or choose_sr(p)
    errors, _ = check_params(p, sr)     # preset texts already explain the warnings
    if errors:
        for e in errors:
            print_wrapped("Error: ", e)
        return
    print_session_header(p, cfg, duration, sr, 0.0 if out == 2 else pretask)

    if out == 2:
        nch = 2
        if p["mode"] == "surround":
            nch = ask_int("  Channels to write (2-16) [6]: ", 2, 16, 6)
        name = "_".join("".join(ch if ch.isalnum() or ch == "." else " "
                                for ch in p["label"].split("--")[0]).split())
        length = f"{round(duration / 60)}min" if duration >= 60 else f"{round(duration)}s"
        dflt = f"{name}_{length}.flac"
        path = input(f"\n  Output file [{dflt}]: ").strip().strip('"') or dflt
        if os.path.splitext(path)[1].lower() not in EXPORT_FORMATS:
            path += ".flac"
        try:
            export_session(p, cfg, duration, sr, path, nch)
        except Exception as e:
            print(f"\n  Error writing {path}: {e}")
        return

    input("\n  Press Enter to begin.  Ctrl+C to stop.\n")
    try:
        elapsed = play_session(p, cfg, duration, sr, pretask)
    except Exception as e:
        print(f"\n  Playback error: {e}")
        return
    if elapsed is None:
        return
    print(f"  Session complete.  Total time: {fmt(elapsed)}")
    ref = p.get("ref_s")
    if ref:
        if elapsed >= ref:
            print(f"  Matched the {fmt(ref)} exposure used in {p['ref_src']}.")
        else:
            print(f"  {fmt(ref - elapsed)} shorter than the {fmt(ref)} exposure used in {p['ref_src']}.")
    print()

# ---------------------------------------------------------------------------
#  SELF-TEST
# ---------------------------------------------------------------------------

def _max_step_bound(p, sr, amp):
    """Largest sample-to-sample step a click-free rendering can contain."""
    f_top = signal_top_hz(p)
    bound = 2 * math.sin(min(math.pi * f_top / sr, math.pi / 2))
    if p["mode"] in ("isochronic", "surround"):
        bound += math.pi / (2 * _pulse_ramp(p["beat_hz"], duty_of(p)) * sr)
    return amp * bound


def run_selftest():
    """Render every preset, then verify frequencies, smoothness and seam-free
    block processing with the analyser.  Returns True if all checks pass."""
    amp, secs, fails = 0.8, 8.0, 0
    print("\nSelf-test: rendering every preset and analysing the result\n")
    for title, suite in SUITES:
        print(f"  {title}")
        for p in suite:
            sr = 96000 if p["mode"] in ("parametric", "hf_am") else SR_DEFAULT
            m, c, b = p["mode"], p["carrier_hz"], p["beat_hz"]
            problems = []
            x = render(p, sr, secs, amplitude=amp, fade_s=0.0)
            res = analyse(x.astype(np.float64), sr)
            ch0 = res["channels"][0]
            tol = lambda f: max(0.02, 1e-4 * f)
            if m == "binaural":
                fl = ch0["peaks"][0][0]
                fr = res["channels"][1]["peaks"][0][0]
                if abs(fl - c) > tol(c) or abs(fr - (c + b)) > tol(c + b):
                    problems.append(f"tones {fl:.3f}/{fr:.3f} Hz, expected {c:g}/{c+b:g}")
                if not res["binaural"] or abs(res["binaural"][2] - b) > 0.02:
                    problems.append(f"binaural offset {res['binaural']}, expected {b:g}")
                if ch0["mod"]:
                    problems.append("unexpected amplitude modulation in a binaural channel")
            elif m in ("monaural", "isochronic", "surround", "hf_am"):
                if not ch0["mod"] or abs(ch0["mod"][0] - b) > tol(b):
                    problems.append(f"modulation {ch0['mod']}, expected {b:g} Hz")
            elif m == "parametric":
                found = [f for f, _ in ch0["peaks"]]
                if not all(any(abs(f - g) < tol(g) for f in found) for g in (c, c + b)):
                    problems.append(f"peaks {found}, expected {c:g} and {c+b:g}")
            else:
                f0 = b if m == "infrasonic" else c
                if abs(ch0["peaks"][0][0] - f0) > tol(f0):
                    problems.append(f"peak {ch0['peaks'][0][0]:.3f} Hz, expected {f0:g}")

            step = float(np.max(np.abs(np.diff(x, axis=0))))
            bound = _max_step_bound(p, sr, amp)
            if step > 1.05 * bound:
                problems.append(f"discontinuity: step {step:.4f} > bound {bound:.4f}")

            kw = dict(amplitude=amp, noise="pink", noise_lvl=0.25, seed=7)
            a = render(p, sr, 2.0, block=BLOCK_SIZE, **kw)
            z = render(p, sr, 2.0, block=997, **kw)
            if a.shape != z.shape or np.max(np.abs(a - z)) > 1e-6:
                problems.append("output depends on block size (seams between blocks)")

            fails += bool(problems)
            print(f"    {'PASS' if not problems else 'FAIL'}  {p['label']}")
            for pr in problems:
                print(f"          {pr}")

    print("\n  Engine")
    p = SUITE_GENERAL[0]
    x = render(p, SR_DEFAULT, 3.0, amplitude=amp, fade_s=1.0)
    ok = abs(x[0]).max() == 0 and abs(x[-1]).max() < 1e-3 and abs(x[len(x)//2]).max() > 0.1
    r = Renderer(p, SR_DEFAULT, amp, fade_s=0.0)
    r.next_block(); r.release()
    rel = []
    while (blk := r.next_block()) is not None:
        rel.append(blk)
    rel = np.concatenate(rel)
    ok_rel = len(rel) == int(STOP_FADE_S * SR_DEFAULT) and abs(rel[-1]).max() < 1e-3
    fails += (not ok) + (not ok_rel)
    print(f"    {'PASS' if ok else 'FAIL'}  fade in / scheduled fade out")
    print(f"    {'PASS' if ok_rel else 'FAIL'}  release fade after stop request")

    with tempfile.TemporaryDirectory() as d:
        src, dst = os.path.join(d, "in.wav"), os.path.join(d, "out.wav")
        t = np.arange(SR_DEFAULT * 6) / SR_DEFAULT
        sf.write(src, 0.9 * np.sin(2 * np.pi * 220 * t), SR_DEFAULT, subtype="PCM_16")
        embed_file(src, dst, 10.0, 0.5)
        y, _ = sf.read(dst, always_2d=True)
        info = sf.info(dst)
        mod = analyse(y, SR_DEFAULT)["channels"][0]["mod"]
        ok = (mod is not None and abs(mod[0] - 10.0) < 0.05 and np.abs(y).max() <= 0.9 + 1e-3
              and info.subtype == "PCM_16")
        fails += not ok
        print(f"    {'PASS' if ok else 'FAIL'}  embed mode (10 Hz AM, no clipping, format kept)")

    print(f"\n  {'All checks passed.' if not fails else f'{fails} check(s) FAILED.'}\n")
    return fails == 0

# ---------------------------------------------------------------------------
#  MAIN
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Brainwave tone generator and audio frequency analyser.")
    ap.add_argument("--detect", metavar="FILE", help="analyse an audio file and exit")
    ap.add_argument("--seconds", type=float, default=60.0, help="seconds to analyse with --detect")
    ap.add_argument("--selftest", action="store_true", help="render and verify every preset")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if run_selftest() else 1)
    if args.detect:
        sys.exit(0 if detect_frequency(args.detect, args.seconds) else 1)

    try:
        choice = choose_suite()
        if choice == 6:
            run_embed(); return
        if choice == 7:
            run_detect(); return
        if choice == 5:
            p = manual_config()
        else:
            title, suite = SUITES[choice - 1]
            p = choose_preset(suite, title)
        run_tone_session(p)
    except (KeyboardInterrupt, EOFError):
        print("\n  Cancelled.")


if __name__ == "__main__":
    main()
