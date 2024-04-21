"""
Brainwave Frequency Generator
=====================================
Requirements
  pip install numpy sounddevice
  pip install scipy   (embed mode only)
"""

import math
import os
import queue
import signal
import sys
import threading
import time

# ---------------------------------------------------------------------------
#  AUTO-INSTALL MISSING PACKAGES
# ---------------------------------------------------------------------------

def _ensure_packages():
    """Install any missing third-party packages automatically."""
    import importlib
    import importlib.util
    import subprocess

    required = {
        "numpy":       "numpy",
        "sounddevice": "sounddevice",
    }

    missing = [
        pip_name
        for mod_name, pip_name in required.items()
        if importlib.util.find_spec(mod_name) is None
    ]

    if missing:
        print(f"  Installing missing package(s): {', '.join(missing)} ...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--break-system-packages",
             "--quiet", *missing]
        )
        print("  Installation complete.\n")

_ensure_packages()

# ---------------------------------------------------------------------------

import numpy as np
import sounddevice as sd

# ---------------------------------------------------------------------------
#  CONSTANTS
# ---------------------------------------------------------------------------

SR_DEFAULT         = 44100
SR_HIGH            = 96000
BLOCK_SIZE         = 4096
PREFILL            = 4
DTYPE              = "float32"
FADE_S             = 4.0
CARRIER_OPTIMAL    = 400.0
SUBLIMINAL_CARRIER = 18500.0
PARAMETRIC_BASE    = 40000.0
BINAURAL_LIMIT     = 30.0

# Research-backed minimum session lengths (seconds).
# Sources: Jirakittayakorn & Wongsawat (2017) Frontiers Neurosci — theta 6 min;
#          Kim et al. (2023) — alpha 5 min;
#          Jirakittayakorn & Wongsawat (2017b) — gamma 15 min;
#          Garcia-Argibay et al. (2018) Psych Research — general 9-10 min optimal;
#          Ingendoh et al. (2023) PLoS ONE — overall review of 14 EEG studies.
MIN = {
    "delta": 1200,   # 20 min  — limited entrainment evidence
    "theta": 600,    # 10 min  — EEG confirmed at parietal/temporal after 6 min
    "alpha": 600,    # 10 min  — EEG confirmed frontal–occipital after 5 min
    "beta":  600,    # 10 min  — zero EEG entrainment confirmed; cognitive effects only
    "gamma": 900,    # 15 min  — EEG confirmed temporal/frontal/central after 15 min
    "other":   0,
}

# Evidence strength codes for display
# "Strong"       — meta-analysis support and/or multiple independent replications
# "Moderate"     — single well-controlled RCT or multiple smaller studies
# "Limited"      — pilot data only or preliminary findings
# "Inconclusive" — no reliable entrainment demonstrated; effect mechanism unknown
EVIDENCE = {
    "gamma_attention":  "Moderate",
    "adhd_15hz":        "Limited",
    "alpha_memory":     "Moderate",
    "beta_memory":      "Moderate",
    "delta":            "Limited",
    "theta":            "Moderate",
    "alpha_relax":      "Moderate",
    "beta_lo":          "Inconclusive",
    "beta_hi":          "Inconclusive",
    "schumann":         "Limited",
}

# ---------------------------------------------------------------------------
#  PRESET DATA
# ---------------------------------------------------------------------------

SUITE_GENERAL = [
    dict(
        label    = "Delta  1.5 Hz  --  Deep sleep / physical recovery",
        mode     = "binaural", beat_hz=1.5, carrier_hz=400.0,
        duty     = 0.5, band="delta", evidence="Limited",
        timing   = "during",
        desc     = """\
    Dominant during NREM stage 3 deep sleep.  Associated with physical
    restoration, immune regulation and growth hormone release.
    Waking delta does not replicate sleep benefits.  Use for sleep onset.
    Headphones required.  Research minimum: 20 min.  Evidence: Limited.""",
        detail   = """\
    Delta oscillations (0.5-3 Hz) are the slowest recorded brainwave pattern
    and occur naturally only during deep, dreamless sleep.  They are associated
    with physical restoration, immune regulation and hormonal release (including
    growth hormone).  Delta is not a conscious state; its presence during
    wakefulness typically indicates drowsiness, pathology, or neurological
    abnormality rather than a beneficial condition.

    Waking delta stimulation does not replicate the restorative effects of
    natural delta sleep.  These are not equivalent processes.  This preset
    is best used as a sleep-onset aid rather than a daytime tool.

    Delivery: binaural beat at 400 Hz carrier (research-optimal per Goodin
    et al.).  Headphones required.  Research minimum: 20 minutes.  Effect
    evidence for waking delta entrainment is limited.
    Evidence: Limited.""",
    ),
    dict(
        label    = "Theta  6 Hz  --  Relaxation / creativity / emotional processing",
        mode     = "binaural", beat_hz=6.0, carrier_hz=400.0,
        duty     = 0.5, band="theta", evidence="Moderate",
        timing   = "before_and_during",
        desc     = """\
    Prominent during light sleep, hypnagogic states and deep meditation.
    Associated with memory consolidation, emotional processing and creative
    ideation.  EEG confirmed at parietal/temporal sites after 6 min.
    Garcia-Argibay (2018): exposure before+during superior to during-only.
    Headphones required.  Research minimum: 10 min.  Evidence: Moderate.""",
        detail   = """\
    Theta oscillations (4-7 Hz) are prominent during light sleep, the hypnagogic
    (pre-sleep) state and deep meditative states.  They are associated with
    memory consolidation, emotional processing and creative ideation.  Elevated
    waking theta in the frontal midline has been correlated with focused internal
    attention in some research contexts.  However, prolonged waking theta can
    also correlate with inattentiveness and drowsiness.

    EEG confirmation: theta power increase at parietal and temporal electrode
    sites after 6 minutes of stimulation at 400 Hz carrier.
    Source: Jirakittayakorn & Wongsawat (2017), Frontiers in Neuroscience.
    400 Hz is the research-optimal carrier for binaural beat perception.

    TIMING NOTE: Garcia-Argibay et al. (2018) meta-analysis of 22 studies found
    exposure before, and before+during the task produced larger effects than
    during-task-only exposure.  Use the "Pre-task induction" duration option.

    Note: theta stimulation during cognitively demanding active tasks may reduce
    rather than enhance performance.  Best used during rest or passive states.

    Delivery: binaural beat.  Headphones required.
    Research minimum: 10 minutes.  Evidence: Moderate.""",
    ),
    dict(
        label    = "Schumann  7.83 Hz  --  Earth electromagnetic resonance",
        mode     = "binaural", beat_hz=7.83, carrier_hz=400.0,
        duty     = 0.5, band="theta", evidence="Limited",
        timing   = "during",
        desc     = """\
    The fundamental resonance frequency of the Earth-ionosphere cavity.
    Falls at the alpha-theta boundary.  Referenced in grounding and
    stress-relief contexts.  Peer-reviewed evidence is limited.
    Headphones required.  Research minimum: 10 minutes.  Evidence: Limited.""",
        detail   = """\
    7.83 Hz is the fundamental resonance frequency of the electromagnetic cavity
    formed between the Earth's surface and the ionosphere.  It falls near the
    alpha-theta boundary of human brainwave activity.

    Some researchers hypothesise a biological entrainment relationship between
    the geomagnetic field and human neural oscillations.  Direct peer-reviewed
    evidence of measurable therapeutic or neurological benefit from 7.83 Hz
    audio stimulation in humans remains limited and inconclusive.

    Chaieb et al. (2015) reviewed ABS literature and found consistent anxiety
    reduction for delta/theta frequencies; Schumann-specific evidence is absent
    from clinical literature.

    Delivery: binaural beat.  Headphones required.
    Research minimum: 10 minutes.  Evidence: Limited.""",
    ),
    dict(
        label    = "Alpha  10 Hz  --  Relaxed wakefulness / stress reduction",
        mode     = "binaural", beat_hz=10.0, carrier_hz=400.0,
        duty     = 0.5, band="alpha", evidence="Moderate",
        timing   = "before_and_during",
        desc     = """\
    Dominant during relaxed, eyes-closed wakefulness and light meditation.
    EEG confirmed increase across frontal, central, parietal and occipital
    alpha power after 5 min.  Kim et al. (2023), alpha 10 Hz, 250 Hz carrier.
    Garcia-Argibay: anxiety reduction confirmed across multiple studies.
    Headphones required.  Research minimum: 10 min.  Evidence: Moderate.""",
        detail   = """\
    Alpha oscillations (8-12 Hz) are the dominant rhythm during relaxed,
    wakeful states such as quiet reflection, closed-eye rest and light
    meditation.  They are associated with reduced cortical arousal, lower
    anxiety and a receptive cognitive state.

    EEG confirmation: increased absolute alpha power across frontal, central,
    parietal and occipital electrode areas after 5 minutes of stimulation
    at 250 Hz carrier.  Source: Kim et al. (2023), Technology and Health Care.

    Garcia-Argibay et al. (2018) meta-analysis (g = 0.69 for anxiety across
    5 studies): theta/delta BB consistently reduce anxiety scores; alpha falls
    at the boundary and shows similar effects.  Kraus & Porubanova (2015) found
    alpha BB (9.55 Hz) improved working memory capacity.

    TIMING NOTE: Exposure before and during the task (Garcia-Argibay 2018)
    produces superior cognitive effects to during-only.

    Alpha increase is a correlate of reduced arousal rather than a direct
    cause of calm.

    Delivery: binaural beat at 400 Hz carrier.  Headphones required.
    Research minimum: 10 minutes.  Evidence: Moderate.""",
    ),
    dict(
        label    = "15 Hz  --  Discomfort threshold  (caution)",
        mode     = "mono", beat_hz=15.0, carrier_hz=15.0,
        duty     = 0.5, band="other", evidence="Inconclusive",
        timing   = "during",
        desc     = """\
    Not recommended for relaxation or general use.  Stimulation near 15 Hz
    has been associated with discomfort, eye strain and headaches in
    sensitive individuals.  Mono delivery.  Restrict to controlled contexts.
    Evidence: Inconclusive.""",
        detail   = """\
    Stimulation near 15 Hz — particularly via visual flicker — has been
    associated with discomfort, eye strain and headaches in sensitive
    individuals.  This is partly explained by proximity to the alpha-beta
    boundary and susceptibility in photosensitive subjects.

    Auditory stimulation at this frequency carries a lower but non-zero risk
    of overstimulation.

    NOTE: 15 Hz binaural beats were used in Malandrone et al. (2022) pilot
    RCT for ADHD with positive subjective studying-performance results.  That
    specific protocol (415/400 Hz binaural beat) is available in the Attention
    & Cognition suite.  This mono preset is for reference completeness only.

    Delivery: mono tone.  No therapeutic use recommended.""",
    ),
    dict(
        label    = "Low Beta  16 Hz  --  Focused attention / logical reasoning",
        mode     = "binaural", beat_hz=16.0, carrier_hz=400.0,
        duty     = 0.5, band="beta", evidence="Inconclusive",
        timing   = "before_and_during",
        desc     = """\
    Low beta (13-20 Hz) correlated with sustained attention and working memory.
    Lane et al. (1998): beta BB improved vigilance vs theta BB.
    CAUTION: zero of 14 peer-reviewed EEG studies (Ingendoh 2023) found
    reliable beta entrainment via binaural beats.  Headphones required.
    Evidence: Inconclusive (for entrainment); Moderate (behavioural effects).""",
        detail   = """\
    The beta band (13-30 Hz) is associated with focused attention, logical
    reasoning, and deliberate information processing via frontoparietal networks.

    Lane et al. (1998) found beta BB (16 and 24 Hz) improved performance on a
    vigilance task and produced less negative mood compared to theta/delta BB.
    This is included in the Garcia-Argibay (2018) meta-analysis which found a
    significant overall effect (g = 0.45) for cognitive outcomes including attention.

    IMPORTANT LIMITATION: Zero of 14 peer-reviewed EEG studies found reliable
    beta-band entrainment via binaural beats (Ingendoh et al. 2023 systematic
    review, PLoS ONE 18(5): e0286023).  Behavioural effects may occur via
    mechanisms other than EEG entrainment (e.g., norepinephrine modulation
    per Hommel et al. 2016).

    Delivery: binaural beat.  Headphones required.
    Research minimum: 10 minutes.  Evidence: Inconclusive (entrainment).""",
    ),
    dict(
        label    = "High Beta  25 Hz  --  Heightened arousal  (caution)",
        mode     = "binaural", beat_hz=25.0, carrier_hz=400.0,
        duty     = 0.5, band="beta", evidence="Inconclusive",
        timing   = "during",
        desc     = """\
    High beta (20-30 Hz) associated with heightened cortical arousal.
    In excess or in stress-prone individuals, correlates with anxiety,
    rumination and overstimulation.  Not recommended for individuals
    prone to anxiety.  Zero EEG studies confirm entrainment at this band.
    Evidence: Inconclusive.""",
        detail   = """\
    High beta oscillations (20-30 Hz) are associated with heightened cortical
    arousal and activation.  In excess or in stress-prone individuals, high beta
    correlates with anxiety, rumination and overstimulation.

    IMPORTANT LIMITATION: Zero of 14 peer-reviewed EEG studies found reliable
    beta-band entrainment via binaural beats (Ingendoh et al. 2023).

    This preset is included for completeness of the beta band coverage.
    It has no recommended therapeutic application.

    Delivery: binaural beat.  Headphones required.  No established minimum.
    Evidence: Inconclusive.""",
    ),
    dict(
        label    = "Gamma  40 Hz  --  High cognition / neural coordination",
        mode     = "isochronic", beat_hz=40.0, carrier_hz=200.0,
        duty     = 0.5, band="gamma", evidence="Moderate",
        timing   = "before_and_during",
        desc     = """\
    Correlated with high-level information processing and distributed neural
    coordination.  40 Hz isochronic used here because 40 Hz exceeds the
    ~30 Hz binaural perception limit.  EEG confirmed in temporal/frontal/
    central regions after 15 min.  Speakers or headphones.
    Research minimum: 15 min.  Evidence: Moderate.""",
        detail   = """\
    Gamma oscillations (40-70 Hz) are correlated with high-level information
    processing, cross-regional neural coordination, perceptual binding, working
    memory maintenance and executive function.  Gamma deficits are documented
    in Alzheimer's disease and schizophrenia.

    40 Hz isochronic is used because this frequency exceeds the approximately
    30 Hz binaural beat perception threshold (Perrott & Nelson 1969).

    EEG confirmation: gamma power increase in temporal, frontal and central
    regions after 15 minutes of stimulation.
    Source: Jirakittayakorn & Wongsawat (2017b), Int J Psychophysiology.

    Engelbregt et al. (2021) demonstrated that 40 Hz binaural beats (440/480 Hz)
    improved Flanker task performance (fewer errors) vs pink noise and monaural
    beats, with large effect sizes.  That binaural variant is in the Attention
    & Cognition suite.

    Delivery: isochronic tone at 200 Hz carrier.  Speakers or headphones.
    Research minimum: 15 minutes.  Evidence: Moderate.""",
    ),
]

SUITE_ATTENTION = [
    dict(
        label    = "40 Hz Attention BB  --  Engelbregt 2021 (440/480 Hz binaural)",
        mode     = "binaural", beat_hz=40.0, carrier_hz=440.0,
        duty     = 0.5, band="gamma", evidence="Moderate",
        timing   = "during",
        desc     = """\
    Replicates the exact parameters from Engelbregt et al. (2021), Experimental
    Brain Research.  Left ear: 440 Hz; Right ear: 480 Hz = perceived 40 Hz beat.
    Reduced Flanker task errors vs pink noise (η²=0.142) with LARGE effect size.
    Monaural beats at same frequencies INCREASED errors — mode matters.
    Headphones required.  Session: 5+ min concurrent with task.  Evidence: Moderate.""",
        detail   = """\
    Source: Engelbregt H, Barmentlo M, Keeser D, Pogarell O, Deijen JB (2021).
    Effects of binaural and monaural beat stimulation on attention and EEG.
    Experimental Brain Research 239:2781-2791. doi:10.1007/s00221-021-06155-z

    STUDY DESIGN: 25 first-year psychology students performed the Eriksen Flanker
    task (3 series of 60 trials) DURING 5-minute presentation of pink noise (PN),
    monaural beats (MB) or binaural beats (BB).  Conditions were counterbalanced
    (within-subject crossover).  EEG was recorded throughout.

    EXACT PARAMETERS: Left ear 440 Hz + right ear 480 Hz = perceived 40 Hz BB.
    Same frequencies presented to both ears simultaneously for MB condition.

    KEY RESULTS (Flanker false responses):
      BB  < PN:  F(1,21)=3.486, p=0.038, η²=0.142  — BB improved accuracy
      MB  > PN:  F(1,21)=18.711, p<0.001, η²=0.471 — MB impaired accuracy
      BB  < MB:  t(23)=1.78, p=0.044 — BB clearly superior to MB

    EEG FINDINGS: No consistent 40/45 Hz power increase was found.  The authors
    conclude the cognitive enhancement is likely not mediated by neural entrainment
    but possibly by norepinephrine/glutamate dynamics (Hommel et al. 2016).

    IMPORTANT: Mode matters critically.  Do not substitute monaural beats for
    binaural beats — the two have opposite effects on attention accuracy.

    TIMING: In this study the task was performed DURING audio exposure.  Use
    standard continuous playback while performing cognitive work.

    Delivery: binaural beat.  Headphones required.
    Recommended session: 5-30 minutes concurrent with task.
    Evidence: Moderate (single well-controlled crossover study, n=24).""",
    ),
    dict(
        label    = "15 Hz ADHD Focus BB  --  Malandrone 2022 (400/415 Hz binaural)",
        mode     = "binaural", beat_hz=15.0, carrier_hz=400.0,
        duty     = 0.5, band="beta", evidence="Limited",
        timing   = "before_and_during",
        desc     = """\
    From Malandrone et al. (2022) pilot add-on RCT, European Psychiatry congress.
    Adult ADHD outpatients: 400 Hz (left) / 415 Hz (right) = 15 Hz beat.
    Significant improvement in subjective studying performance vs placebo
    (mean diff=2.7, p<0.001).  No significant change in ADHD-RS or SART.
    Self-administered during individual study sessions.  Min: 10 min.
    Evidence: Limited (pilot, small sample, conference proceedings).""",
        detail   = """\
    Source: Malandrone F, Spadotto M, Boero M, Bracco IF, Oliva F (2022).
    A pilot add-on Randomized-Controlled Trial evaluating the effect of binaural
    beats on study performance, mind-wandering, and core symptoms of adult ADHD.
    European Psychiatry 65(Suppl1):S274. doi:10.1192/j.eurpsy.2022.701

    STUDY DESIGN: University students with pharmacologically treated adult ADHD
    in a two-group RCT.  Intervention: 15 Hz BB (415 Hz right ear, 400 Hz left
    ear).  Placebo: identical track with two identical 400 Hz tones (no beat).
    Baseline (T0) + two fortnightly follow-ups (T1, T2).

    OUTCOME MEASURES:
      ADHD-RS-5 (rating scale) — no significant change
      MEWS (mind-wandering scale) — no significant change
      SART (sustained attention) — no significant change
      SSP (subjective studying performance) — SIGNIFICANT improvement in BB group
        only: mean difference = 2.7, p < 0.001; between-group contrast at T3.

    INTERPRETATION: The improvement was in subjective study experience rather than
    objective attention metrics.  This may reflect reduced discomfort, anxiety or
    effort perception during studying rather than direct attentional enhancement.
    Chaieb et al. (2015) note that ADHD binaural beat research showed participants
    reported subjectively fewer inattention problems despite no objective change.

    SELF-ADMINISTRATION: Participants used this during their own study sessions.
    Recommend playing continuously while working or studying.

    LIMITATIONS: Pilot study; conference proceedings; small sample; no EEG
    measurement; pharmacological treatment confound.

    Delivery: binaural beat.  Headphones required.
    Recommended: played during self-directed study or focused work.
    Research minimum: 10 minutes.  Evidence: Limited.""",
    ),
    dict(
        label    = "Alpha  9.55 Hz  --  Working memory (Kraus & Porubanová 2015)",
        mode     = "binaural", beat_hz=9.55, carrier_hz=400.0,
        duty     = 0.5, band="alpha", evidence="Moderate",
        timing   = "before",
        desc     = """\
    Kraus & Porubanová (2015): alpha BB improved working memory capacity on
    automated OSPAN task.  12 min exposure before task.  Sea-sound masking used.
    Garcia-Argibay (2018) meta-analysis: exposure BEFORE task yields larger
    effects than during-only.  Use the pre-task induction timer option.
    Headphones required.  Induction: 12 min.  Evidence: Moderate.""",
        detail   = """\
    Source: Kraus J & Porubanová M (2015). The effect of binaural beats on
    working memory capacity.  Studia Psychologica 57(2):135.
    Also cited in Garcia-Argibay et al. (2018) meta-analysis (Table 1, ID 22,
    g = 0.681).

    STUDY DESIGN: 20 participants per group; alpha BB at 9.55 Hz during 12
    minutes of OSPAN (Automated Operation Span Task) performance.  Sea-sound
    background masking used.  Working memory measured as correctly recalled items.

    RESULT: Significant improvement in working memory capacity in the BB group
    compared to control.

    TIMING NOTE: Garcia-Argibay et al. (2018) meta-regression found that exposure
    BEFORE the task (b=0.46, p=0.006) and BEFORE+DURING (b=0.53, p=0.002) both
    produced significantly larger effects than during-task-only exposure.

    MASKING: The original study used sea-sound background.  Pink noise masking
    (available as an option at startup) is the closest validated equivalent.
    Garcia-Argibay found pink noise masking does not reduce effectiveness.

    Delivery: binaural beat at 400 Hz carrier.  Headphones required.
    Recommended: 12 min induction before beginning memory/study tasks.
    Evidence: Moderate (single study; included in meta-analysis).""",
    ),
    dict(
        label    = "Beta  13 Hz  --  Memory encoding (Garcia-Argibay 2017)",
        mode     = "binaural", beat_hz=13.0, carrier_hz=400.0,
        duty     = 0.5, band="beta", evidence="Moderate",
        timing   = "before_and_during",
        desc     = """\
    Beta BB during encoding phase improves free recall and recognition (d'index).
    Garcia-Argibay et al. (2017): 17 min BB before+during encoding.  g=0.907 for
    free recall, g=1.501 for recognition sensitivity.  Best with white/pink noise.
    Garcia-Argibay 2018 meta-analysis: before+during > during-only.
    Headphones required.  Recommended: 17 min before+during encoding.
    Evidence: Moderate.""",
        detail   = """\
    Source: Garcia-Argibay M, Santed MA, Reales JM (2017). Binaural auditory beats
    affect long-term memory.  Psychological Research (advance online publication).
    doi:10.1007/s00426-017-0959-2
    Also in Garcia-Argibay et al. (2018) meta-analysis Table 1, IDs 13-16.

    STUDY DESIGN: 32 participants; 17-minute beta BB (white noise masked) during
    the encoding phase of a free recall and recognition memory task.

    RESULTS (Hedges' g):
      Beta BB — free recall:        g = 0.907
      Beta BB — recognition (d'):   g = 1.501
      Theta BB — free recall:       g = -0.819  (theta IMPAIRED recall)
      Theta BB — recognition:       g = -0.526  (theta IMPAIRED recognition)

    CRITICAL FINDING: Beta improves memory encoding; theta impairs it.
    Direction of effect depends critically on the frequency chosen.

    TIMING: 17 minutes before and during the encoding phase (not the recall phase).
    Use pre-task induction timer: set 5 min pre-task, then continuous during study.

    MASKING: White noise used in original study.  Pink noise masking (available
    at startup) is an acceptable equivalent per Garcia-Argibay 2018.

    Delivery: binaural beat.  Headphones required.
    Recommended: 17 min beginning before and continuing during encoding.
    Evidence: Moderate (single RCT, large effect; included in meta-analysis).""",
    ),
    dict(
        label    = "Gamma  40 Hz  --  Attentional focus (Colzato 2017, global-local)",
        mode     = "isochronic", beat_hz=40.0, carrier_hz=200.0,
        duty     = 0.5, band="gamma", evidence="Moderate",
        timing   = "before_and_during",
        desc     = """\
    Colzato et al. (2017): 3 min gamma BB before+during global-local task
    produced more attentional focus (reduced global precedence effect).
    Garcia-Argibay meta: attention g=0.694 for Colzato; g=0.666 for Hommel 2016.
    Speakers or headphones.  Recommended: 3-10 min induction.
    Research minimum: 15 min.  Evidence: Moderate.""",
        detail   = """\
    Sources:
    Colzato LS, Barone H, Sellaro R, Hommel B (2017). More attentional focusing
    through binaural beats: evidence from the global-local task.  Psychological
    Research 81:271-277.  doi:10.1007/s00426-015-0727-0
    Hommel B, Sellaro R, Fischer R, Borg S, Colzato LS (2016). High-frequency
    binaural beats increase cognitive flexibility.  Frontiers in Psychology 7:1287.
    Garcia-Argibay et al. (2018) meta-analysis, attention subgroup: g = 0.58
    [0.34, 0.83], zero heterogeneity, p < 0.001.

    COLZATO 2017 DESIGN: 36 healthy adults; gamma BB (white noise, 3 min) before
    and during a global-local task.  Compared to constant 340 Hz tone control.
    Result: gamma BB produced more focused attentional style (less global
    precedence); g = 0.694.

    HOMMEL 2016: Gamma BB promoted cognitive flexibility (distributed parallel
    processing style); g = 0.666.

    NOTE: Colzato found narrowed attention; Hommel found increased flexibility.
    These findings appear contradictory but may reflect different task demands.
    Both show gamma BB modulates attentional style.

    FORMAT: 40 Hz exceeds the ~30 Hz binaural perception limit; isochronic AM
    is used for reliability.  Original studies used binaural + white noise.

    Delivery: isochronic tone at 200 Hz carrier.  Speakers or headphones.
    Recommended: 3+ min before, then during cognitive task.
    Research minimum: 15 minutes.  Evidence: Moderate.""",
    ),
]

SUITE_MINDWAR = [
    dict(
        label    = "Delta  2 Hz  --  Deep sleep / cognitive suppression",
        mode     = "binaural", beat_hz=2.0, carrier_hz=400.0,
        duty     = 0.5, band="delta", evidence="Limited",
        timing   = "during",
        desc     = """\
    MindWar characterisation: "1-3 Hz, characteristic of deep sleep."
    Aquino describes delta BWR as making complex or creative effort
    "exhausting and fruitless."  Consistent with standard neuroscience.
    Headphones required.  Research minimum: 20 minutes.  Evidence: Limited.""",
        detail   = """\
    Source: Aquino, MindWar 2nd ed. (2016), Chapter 3, PSYCON #2: Brainwave
    Resonance.

    MindWar characterisation: "1-3 Hz = delta waves, characteristic of deep
    sleep."  Aquino describes the operational use of delta BWR as suppressing
    beta activity in a target population, making anything complex or creative
    "exhausting and fruitless."

    This characterisation is consistent with standard neuroscience.  Delta is
    the dominant oscillatory pattern during NREM stage 3 sleep and is not
    associated with productive waking cognition.

    Delivery: binaural beat.  Headphones required.
    Research minimum: 20 minutes.  Evidence: Limited.""",
    ),
    dict(
        label    = "Theta  5.5 Hz  --  Emotional arousal / frustration",
        mode     = "binaural", beat_hz=5.5, carrier_hz=400.0,
        duty     = 0.5, band="theta", evidence="Limited",
        timing   = "during",
        desc     = """\
    MindWar characterisation: "4-7 Hz, characteristic of high emotion,
    violence and frustration."  This diverges from mainstream research,
    which associates theta with relaxation and meditative states.
    Headphones required.  Research minimum: 10 minutes.  Evidence: Limited.""",
        detail   = """\
    Source: Aquino, MindWar 2nd ed. (2016), Chapter 3, PSYCON #2.

    MindWar characterisation: "4-7 Hz = theta waves, characteristic of high
    emotion, violence and frustration."  In Aquino's framework, theta BWR
    is used to promote emotional volatility and impair rational decision-making
    in a target population.

    DIVERGES FROM CONSENSUS: This framing conflicts with mainstream peer-reviewed
    EEG literature, which primarily associates theta with relaxation, memory
    consolidation and meditative states.  Garcia-Argibay et al. (2017) found
    theta BB actually IMPAIRED memory encoding (g = -0.819).

    Delivery: binaural beat.  Headphones required.
    Research minimum: 10 minutes.  Evidence: Limited.""",
    ),
    dict(
        label    = "Alpha  10 Hz  --  Relaxed / cooperative state",
        mode     = "binaural", beat_hz=10.0, carrier_hz=400.0,
        duty     = 0.5, band="alpha", evidence="Moderate",
        timing   = "during",
        desc     = """\
    MindWar characterisation: "8-12 Hz, characteristic of meditation,
    relaxation and searching for patterns."  Aquino describes alpha BWR
    as enabling "relaxed, pleasant, cooperative discussion."  Consistent
    with mainstream EEG literature.  Headphones required.  Min: 10 minutes.
    Evidence: Moderate.""",
        detail   = """\
    Source: Aquino, MindWar 2nd ed. (2016), Chapter 3, PSYCON #2.

    MindWar characterisation: "8-12 Hz = alpha waves, characteristic of
    meditation, relaxation and searching for patterns."  Consistent with
    mainstream EEG literature.  EEG confirmed: Kim et al. (2023).

    Delivery: binaural beat at 400 Hz carrier.  Headphones required.
    Research minimum: 10 minutes.  Evidence: Moderate.""",
    ),
    dict(
        label    = "Beta  17 Hz  --  Deliberate effort / logical thought",
        mode     = "mono", beat_hz=17.0, carrier_hz=17.0,
        duty     = 0.5, band="beta", evidence="Inconclusive",
        timing   = "during",
        desc     = """\
    MindWar characterisation: "13-22 Hz, frontal brain activity, deliberate
    effort, logical thought."  BWR Manual flags 17 Hz for restlessness and
    anxiety risk.  Best format per BWR Manual: Mono.
    No peer-reviewed EEG entrainment support.  Evidence: Inconclusive.""",
        detail   = """\
    Source: Aquino, MindWar 2nd ed. (2016), Chapter 3, PSYCON #2; BWR Manual.

    CAUTION: 17 Hz is flagged in the BWR Manual for restlessness, discomfort,
    and anxiety risk in sensitive individuals.

    Zero of 14 peer-reviewed EEG studies confirmed reliable beta entrainment
    via binaural beats (Ingendoh et al. 2023).

    Not recommended for individuals prone to anxiety.
    Delivery: mono tone.  No established research minimum.
    Evidence: Inconclusive.""",
    ),
    dict(
        label    = "Infrasonic  12.5 Hz  --  Covert subliminal vector (subliminal mode)",
        mode     = "subliminal", beat_hz=12.5, carrier_hz=SUBLIMINAL_CARRIER,
        duty     = 0.5, band="alpha", evidence="Limited",
        timing   = "during",
        desc     = """\
    MindWar's primary SLIPC delivery vector.  10-15 Hz described as "too low
    to be consciously detected but capable of inducing resonance in the brain."
    This preset AM-modulates an 18,500 Hz carrier at 12.5 Hz.  The carrier
    may be inaudible.  Recommended: 96 kHz sample rate.  Min: 10 minutes.
    Evidence: Limited.""",
        detail   = """\
    Source: Aquino, MindWar 2nd ed. (2016), Chapter 3, PSYCON #1 and #2.

    LIMITATION: MindWar's true intended delivery is ELF electromagnetic fields
    transmitted through the environment, penetrating walls and the body's skin
    surface.  Acoustic software cannot replicate this.

    Delivery: AM subliminal.  Speakers or headphones.  96 kHz recommended.
    Research minimum: 10 minutes.  Evidence: Limited.""",
    ),
    dict(
        label    = "Infrasonic  12.5 Hz  --  Skin-surface pressure wave (infrasonic mode)",
        mode     = "infrasonic", beat_hz=12.5, carrier_hz=12.5,
        duty     = 0.5, band="alpha", evidence="Limited",
        timing   = "during",
        desc     = """\
    Outputs a true 12.5 Hz pressure wave with no audible carrier.  The wave
    is felt through skin and bone conduction rather than heard.  Closest
    acoustic equivalent to Aquino's skin-surface coupling mechanism.
    Requires a subwoofer capable of sub-20 Hz output.  Min: 10 minutes.
    Evidence: Limited.""",
        detail   = """\
    Source: Aquino, MindWar 2nd ed. (2016), Chapter 3, PSYCON #1 and #2.

    This preset outputs a real 12.5 Hz pressure wave with no audible carrier
    tone.  At this frequency the wave is not consciously heard but is felt
    through skin and bone conduction across the entire body surface.

    HARDWARE REQUIRED: Most consumer speakers physically cannot move sufficient
    air at 12.5 Hz.  A subwoofer rated for sub-20 Hz output is required.

    LIMITATION: Acoustic pressure waves are not electromagnetic fields.
    They will not penetrate walls.

    Delivery: direct infrasonic output.  Sub-20 Hz subwoofer required.
    Research minimum: 10 minutes.  Evidence: Limited.""",
    ),
    dict(
        label    = "ELF  57.5 Hz  --  Project Sanguine / biological hazard range",
        mode     = "isochronic", beat_hz=57.5, carrier_hz=200.0,
        duty     = 0.5, band="other", evidence="Inconclusive",
        timing   = "during",
        desc     = """\
    Center of the 45-70 Hz range used by the U.S. Navy's Project Sanguine
    ELF submarine transmitter.  Aquino cites Becker: these frequencies
    "alter blood chemistry, blood pressure and brain wave patterns."
    Speakers or headphones.  No established research minimum.
    Evidence: Inconclusive.""",
        detail   = """\
    Source: Aquino, MindWar 2nd ed. (2016), Chapter 3, PSYCON #1; Becker via
    Aquino.

    LIMITATION: True ELF effects require electromagnetic antenna hardware.
    This preset is an isochronic acoustic output only.

    Delivery: isochronic tone.  Speakers or headphones.  No established minimum.
    Evidence: Inconclusive.""",
    ),
]

SUITE_ALZHEIMERS = [
    dict(
        label    = "MIT GENUS  40 Hz  --  Full 1-hour daily protocol",
        mode     = "isochronic", beat_hz=40.0, carrier_hz=200.0,
        duty     = 0.5, band="gamma", evidence="Moderate",
        timing   = "during",
        desc     = """\
    MIT Picower Institute (Tsai lab) protocol.  40 Hz isochronic tone
    replicating the LED-flicker GENUS method.  Sound alone reduced amyloid
    and tau in Alzheimer's mouse models.  Combined light and sound was most
    effective.  Phase III human trials ongoing (Cognito Therapeutics).
    Speakers or headphones.  Recommended: 60 minutes daily.  Evidence: Moderate.""",
        detail   = """\
    Source: Iaccarino et al. (2016) Nature; MIT Picower Institute / Tsai Lab.

    The MIT Picower Institute (Li-Huei Tsai lab) demonstrated from 2016 onward
    that 40 Hz GENUS (Gamma Entrainment Using Sensory Stimuli) light flickering
    and/or sound clicking reduced amyloid-beta plaques and tau tangles in
    transgenic Alzheimer's disease mouse models, prevented neuron death and
    improved learning and memory.

    Sound alone (without light) also reduced amyloid and tau and improved
    cognition in mice.  Combined light and sound was most effective.  The
    proposed mechanism involves:
      - Increased gamma neural network connectivity
      - Widened blood vessels and glymphatic channels
      - Enhanced CSF-driven waste clearance (amyloid flushed from brain)
      - Reduced microglial inflammatory activity
      - Gene expression changes in neurons and glia

    In subsequent human pilot studies, 40 Hz stimulation was found safe,
    increased gamma-band activity and connectivity and showed apparent benefit
    in early-stage Alzheimer's patients.  MIT spin-off Cognito Therapeutics has
    advanced this into Phase III clinical trials.

    Human clinical efficacy is not yet confirmed at Phase III level.

    FORMAT: Isochronic used because 40 Hz exceeds the ~30 Hz binaural perception
    threshold (Perrott & Nelson 1969).

    Delivery: isochronic tone at 200 Hz carrier.  Speakers or headphones.
    Protocol: 1 hour daily.  Research minimum: 15 minutes.  Evidence: Moderate.""",
    ),
    dict(
        label    = "40 Hz Binaural  --  ASSR via headphones (440/480 Hz)",
        mode     = "binaural", beat_hz=40.0, carrier_hz=440.0,
        duty     = 0.5, band="gamma", evidence="Moderate",
        timing   = "during",
        desc     = """\
    40 Hz binaural beat using Engelbregt 2021 carrier (440/480 Hz).  40 Hz
    exceeds the binaural perception limit but an ASSR is still measurable in
    EEG.  Also showed attention improvement vs pink noise in Flanker task.
    Headphones required.  Research minimum: 15 minutes.  Evidence: Moderate.""",
        detail   = """\
    Sources: Jirakittayakorn & Wongsawat (2017b); Schwarz & Taylor (2005);
    Engelbregt et al. (2021) Experimental Brain Research.

    This preset uses the 440/480 Hz carrier pair from Engelbregt et al. (2021),
    which demonstrated both attention improvement (Flanker task) and is within
    the range used by Schwarz & Taylor (2005) who confirmed 40 Hz ASSR at 380/420
    and 390/430 Hz.  Grose & Mamo (2012) showed frequency-following responses
    at 390/430 Hz.

    40 Hz binaural ASSR confirmed by Schwarz & Taylor (2005) and Jirakittayakorn
    & Wongsawat (2017b).

    Delivery: binaural beat.  Headphones required.
    Research minimum: 15 minutes.  Evidence: Moderate.""",
    ),
    dict(
        label    = "40 Hz Daily Maintenance  --  Short session (15 min)",
        mode     = "isochronic", beat_hz=40.0, carrier_hz=200.0,
        duty     = 0.5, band="gamma", evidence="Moderate",
        timing   = "during",
        desc     = """\
    Short daily session preset for regular use within a sustained protocol.
    Ismail et al. (2018) found 10-day human trials insufficient for amyloid
    reduction, indicating weeks or months of consistent daily sessions are
    likely required for measurable effect.
    Speakers or headphones.  Research minimum: 15 minutes.  Evidence: Moderate.""",
        detail   = """\
    Source: Tsai Lab; Ismail et al. (2018), International Journal of
    Alzheimer's Disease.

    Ismail et al. (2018) found that 10 days was insufficient for amyloid
    reduction.  This preset is designed for daily short sessions as part of
    a sustained longer-term protocol rather than as a standalone treatment.

    Delivery: isochronic tone at 200 Hz carrier.  Speakers or headphones.
    Research minimum: 15 minutes.  Evidence: Moderate.""",
    ),
    dict(
        label    = "40 Hz Parametric  --  Ultrasonic in-air beat",
        mode     = "parametric", beat_hz=40.0, carrier_hz=PARAMETRIC_BASE,
        duty     = 0.5, band="gamma", evidence="Inconclusive",
        timing   = "during",
        desc     = """\
    Two ultrasonic carriers at 40,000 Hz and 40,040 Hz.  Neither is audible.
    Their interaction produces a 40 Hz AM envelope in the air itself via
    nonlinear acoustic demodulation.
    Requires 96 kHz interface and a transducer passing above 40 kHz.
    Evidence: Inconclusive (no clinical data for this delivery format).""",
        detail   = """\
    Source: Parametric audio principle applied to MIT Picower 40 Hz target.

    Two ultrasonic carrier tones are generated at 40,000 Hz and 40,040 Hz.
    Neither frequency is audible to humans.  Their amplitude-modulated sum
    creates a 40 Hz envelope in the ultrasonic band.

    HARDWARE REQUIREMENT: Most consumer audio interfaces and DACs roll off
    well before 40 kHz.  A professional interface with flat response to 40+
    kHz and a compatible transducer are required.

    Delivery: parametric ultrasonic.  96 kHz SR required.
    Evidence: Inconclusive (delivery format untested in clinical trials).""",
    ),
    dict(
        label    = "Upper Gamma  60 Hz  --  High cognition / distributed coordination",
        mode     = "isochronic", beat_hz=60.0, carrier_hz=200.0,
        duty     = 0.5, band="gamma", evidence="Inconclusive",
        timing   = "during",
        desc     = """\
    Upper gamma range (40-70 Hz), correlated with perceptual binding,
    distributed cortical coordination and intense attentional states.
    Auditory gamma entrainment research is less established than 40 Hz.
    Speakers or headphones.  Research minimum: 15 minutes.
    Evidence: Inconclusive (insufficient research at 60 Hz specifically).""",
        detail   = """\
    60 Hz sits within the 40-70 Hz gamma range correlated with high-level
    information processing, perceptual binding, cross-regional neural
    coordination and states of intense attentional focus.

    Auditory gamma entrainment research is considerably less established than
    visual (LED flicker) GENUS research.  The 40 Hz target has by far the
    strongest evidence base.  Individual entrainment responses at 60 Hz are
    highly variable.

    Delivery: isochronic tone at 200 Hz carrier.  Speakers or headphones.
    Research minimum: 15 minutes.
    Evidence: Inconclusive (40 Hz data does not extrapolate to 60 Hz).""",
    ),
]

# ---------------------------------------------------------------------------
#  GLOBAL STATE
# ---------------------------------------------------------------------------

stop_event = threading.Event()
_t0_wall   = None
_amplitude = 0.8
_pink_lvl  = 0.0   # 0.0 = no pink noise, >0 = blend level

# ---------------------------------------------------------------------------
#  SIGNAL GENERATORS  (vectorised — no per-sample loops)
# ---------------------------------------------------------------------------

def _env(t0, n, fi, fo, tot):
    e = np.ones(n, dtype=np.float32)
    if fi > 0 and t0 < fi:
        hi = min(t0+n, fi); k = hi-t0
        e[:k] = np.linspace(t0/fi, hi/fi, k, endpoint=False, dtype=np.float32)
    if tot and fo > 0:
        fs = tot-fo
        if t0+n > fs:
            lb  = max(0, fs-t0)
            a0  = max(0.0, (tot-(t0+lb))/fo)
            a1  = max(0.0, (tot-(t0+n))/fo)
            e[lb:] *= np.linspace(a0, a1, n-lb, endpoint=False, dtype=np.float32)
    return e

def _t(t0, n, sr):
    return (np.arange(n, dtype=np.float64)+t0)/sr

def _gate(t, beat, duty, sr, n):
    gate = ((t*beat)%1.0 < duty).astype(np.float32)
    tl   = min(int(sr*0.005), max(1, n//8))
    if tl > 1:
        tap  = np.hanning(tl*2)[:tl].astype(np.float32)
        diff = np.diff(gate, prepend=gate[0])
        for i in np.where(diff>0)[0]:
            j=min(i+tl,n); gate[i:j]*=tap[:j-i]
        for i in np.where(diff<0)[0]:
            j=min(i+tl,n); gate[i:j]*=tap[:j-i][::-1]
    return gate

def gen_pink_noise(n):
    """Pink noise via FFT spectral shaping (1/sqrt(f)), Garcia-Argibay 2018:
    pink noise masking does not reduce binaural beat efficacy; Chaieb 2015:
    may amplify the binaural beat percept."""
    white = np.random.randn(n)
    fft   = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = 1.0                         # avoid div-by-zero at DC
    fft      = fft / np.sqrt(freqs)
    pink     = np.fft.irfft(fft, n)
    peak     = np.max(np.abs(pink))
    return (pink / peak if peak > 0 else pink).astype(np.float32)

def stereo(m):
    return np.column_stack((m, m))

def _apply_pink(stereo_blk, n):
    """Blend pink noise into a stereo block according to global _pink_lvl."""
    if _pink_lvl <= 0.0:
        return stereo_blk
    pink  = gen_pink_noise(n)
    pink2 = np.column_stack((pink, pink))
    return stereo_blk * (1.0 - _pink_lvl) + pink2 * _pink_lvl

def gen_binaural(carr, beat, t0, n, sr, fi, fo, tot):
    t = _t(t0, n, sr); e = _env(t0, n, fi, fo, tot)
    L = (np.sin(2*np.pi*carr*t)*e).astype(np.float32)
    R = (np.sin(2*np.pi*(carr+beat)*t)*e).astype(np.float32)
    return np.column_stack((L, R))

def gen_isochronic(carr, beat, t0, n, sr, duty, fi, fo, tot):
    t = _t(t0, n, sr); e = _env(t0, n, fi, fo, tot)
    m = (np.sin(2*np.pi*carr*t)*_gate(t, beat, duty, sr, n)*e).astype(np.float32)
    return stereo(m)

def gen_mono(carr, t0, n, sr, fi, fo, tot):
    t = _t(t0, n, sr); e = _env(t0, n, fi, fo, tot)
    return stereo((np.sin(2*np.pi*carr*t)*e).astype(np.float32))

def gen_infrasonic(beat, t0, n, sr, fi, fo, tot):
    return gen_mono(beat, t0, n, sr, fi, fo, tot)

def gen_subliminal(carr, beat, t0, n, sr, fi, fo, tot):
    t  = _t(t0, n, sr); e = _env(t0, n, fi, fo, tot)
    am = (1.0+np.cos(2*np.pi*beat*t))/2.0
    return stereo((np.sin(2*np.pi*carr*t)*am*e).astype(np.float32))

def gen_parametric(base, beat, t0, n, sr, fi, fo, tot):
    t = _t(t0, n, sr); e = _env(t0, n, fi, fo, tot)
    w = ((np.sin(2*np.pi*base*t)+np.sin(2*np.pi*(base+beat)*t))*0.5*e).astype(np.float32)
    return stereo(w)

def gen_surround(carr, beat, t0, n, sr, nch, duty, fi, fo, tot):
    blk = gen_isochronic(carr, beat, t0, n, sr, duty, fi, fo, tot)
    return np.tile(blk[:,0:1], (1, nch))

def norm(b):
    p = np.max(np.abs(b))
    b = b/p if p > 1.0 else b
    return (b * _amplitude).astype(np.float32)

# ---------------------------------------------------------------------------
#  PRODUCER + PLAYBACK
# ---------------------------------------------------------------------------

def producer(p, q, stop_ev, sr, total=None, nch=2):
    fi = int(FADE_S*sr); t0 = 0
    while not stop_ev.is_set():
        if total and t0 >= total:
            q.put(None); break
        fill = BLOCK_SIZE if not total else min(BLOCK_SIZE, total-t0)
        kw   = dict(t0=t0, n=fill, sr=sr, fi=fi, fo=fi, tot=total)
        m    = p["mode"]
        if   m=="binaural":   blk=gen_binaural(p["carrier_hz"],p["beat_hz"],**kw)
        elif m=="isochronic": blk=gen_isochronic(p["carrier_hz"],p["beat_hz"],duty=p["duty"],**kw)
        elif m=="mono":       blk=gen_mono(p["carrier_hz"],**kw)
        elif m=="infrasonic": blk=gen_infrasonic(p["beat_hz"],**kw)
        elif m=="subliminal": blk=gen_subliminal(p["carrier_hz"],p["beat_hz"],**kw)
        elif m=="parametric": blk=gen_parametric(p["carrier_hz"],p["beat_hz"],**kw)
        elif m=="surround":   blk=gen_surround(p["carrier_hz"],p["beat_hz"],
                                               t0,fill,sr,nch,p["duty"],fi,fi,total)
        else:                 blk=gen_mono(p["carrier_hz"],**kw)
        blk = _apply_pink(blk, fill)
        blk = norm(blk)
        if fill < BLOCK_SIZE:
            blk = np.vstack((blk, np.zeros((BLOCK_SIZE-fill,blk.shape[1]),dtype=np.float32)))
        q.put(blk); t0 += fill

def make_cb(q):
    def cb(out, frames, ti, st):
        try:   blk = q.get_nowait()
        except queue.Empty: out[:] = 0; return
        if blk is None: raise sd.CallbackStop()
        out[:] = blk.reshape(out.shape)
    return cb

def prefill(q, n=PREFILL//2):
    while q.qsize() < n: pass

def get_nch():
    try:
        d = sd.query_devices(sd.default.device[1], 'output')
        return max(2, int(d['max_output_channels']))
    except: return 2

def fmt(s):
    s = int(s); h,r = divmod(s,3600); m,sc = divmod(r,60)
    return f"{h:02d}:{m:02d}:{sc:02d}"

# Research milestone markers (seconds) for progress display
_MILESTONES = [
    (300,  "5 min  — alpha/theta minimum (Kim 2023)"),
    (360,  "6 min  — theta EEG confirmed (Jirakittayakorn 2017)"),
    (600,  "10 min — optimal cortical spread (Garcia-Argibay 2018)"),
    (900,  "15 min — gamma EEG confirmed (Jirakittayakorn 2017b)"),
    (1020, "17 min — memory encoding window (Garcia-Argibay 2017)"),
    (1200, "20 min — delta minimum"),
    (3600, "60 min — MIT GENUS full protocol"),
]
_milestones_fired = set()

def progress_thread(min_s, total_s):
    global _t0_wall
    _t0_wall = time.time()
    _milestones_fired.clear()
    while not stop_event.is_set():
        e   = time.time() - _t0_wall
        pct = min(e/min_s*100, 100) if min_s > 0 else 100
        bar = "="*int(pct/5) + "-"*(20-int(pct/5))
        rem = f"  rem {fmt(max(0, total_s-e))}" if total_s else "  continuous"
        sys.stdout.write(f"\r  {fmt(e)}  [{bar}] {pct:5.1f}%{rem}   ")
        sys.stdout.flush()
        for ts, label in _MILESTONES:
            if e >= ts and ts not in _milestones_fired:
                _milestones_fired.add(ts)
                sys.stdout.write(f"\n  *** MILESTONE: {label}\n  ")
                sys.stdout.flush()
        time.sleep(1)
    sys.stdout.write("\n")

def run_session(p, duration, sr, pretask=0.0):
    """Run audio session.  pretask = pre-task induction seconds (then alert)."""
    global _t0_wall
    nch   = get_nch() if p["mode"]=="surround" else 2
    total = None if not duration else int(duration*sr)
    q     = queue.Queue(maxsize=PREFILL*2)
    local = threading.Event()
    sev   = local if total else stop_event

    thr = threading.Thread(target=producer, args=(p,q,sev,sr,total,nch), daemon=True)
    thr.start(); prefill(q)
    threading.Thread(target=progress_thread,
                     args=(MIN.get(p["band"],0), duration if duration else None),
                     daemon=True).start()

    if pretask > 0:
        # Start a thread to fire the pre-task alert
        def _pretask_alert():
            time.sleep(pretask)
            if not stop_event.is_set():
                sys.stdout.write(
                    f"\n\n  ╔══════════════════════════════════╗\n"
                    f"  ║  >>> BEGIN YOUR TASK NOW <<<     ║\n"
                    f"  ║  Pre-task induction complete      ║\n"
                    f"  ╚══════════════════════════════════╝\n\n  ")
                sys.stdout.flush()
        threading.Thread(target=_pretask_alert, daemon=True).start()

    ch = nch if p["mode"]=="surround" else 2
    with sd.OutputStream(samplerate=sr, channels=ch,
                         blocksize=BLOCK_SIZE, dtype=DTYPE, callback=make_cb(q)):
        if total:
            sd.sleep(int(duration*1000)); local.set()
        else:
            stop_event.wait()
    thr.join(timeout=2)
    if total:
        stop_event.set()

# ---------------------------------------------------------------------------
#  EMBED MODE
# ---------------------------------------------------------------------------

def run_embed():
    W = 64
    print()
    print("="*W)
    print("  Embed Mode -- AM-modulate an existing WAV file")
    print("="*W)
    print()
    print("  Loads a WAV file and applies amplitude modulation at the")
    print("  chosen beat frequency.  The output file is indistinguishable")
    print("  from the source on casual listening while carrying the AM")
    print("  envelope in its waveform amplitude.")
    print()
    print("  Source: Aquino MindWar 'insertion into electronic media' vector.")
    print()
    inp = input("  Input WAV path  : ").strip().strip('"')
    if not os.path.isfile(inp):
        print("  Error: file not found."); return
    out  = input("  Output WAV path : ").strip().strip('"')
    while True:
        try:
            beat = float(input("  Beat frequency (Hz) : ").strip())
            if beat > 0: break
        except ValueError: pass
    val = input("  Modulation depth 0.0-1.0  [0.15 = subtle / 1.0 = full] : ").strip()
    try:    depth = max(0.0, min(1.0, float(val) if val else 0.15))
    except: depth = 0.15

    try:
        try:
            import importlib
            import importlib.util
            if importlib.util.find_spec("scipy") is None:
                print("  Installing missing package: scipy ...")
                import subprocess
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install",
                     "--break-system-packages", "--quiet", "scipy"]
                )
                print("  Installation complete.\n")
        except Exception as _install_err:
            print(f"  Error installing scipy: {_install_err}"); return
        from scipy.io import wavfile

        sr, data = wavfile.read(inp)
        orig = data.dtype
        if np.issubdtype(orig, np.integer):
            data = data.astype(np.float32) / np.iinfo(orig).max
        else:
            data = data.astype(np.float32)

        t   = np.arange(data.shape[0], dtype=np.float64) / sr
        env = 1.0 + depth * np.sin(2*np.pi*beat*t)
        mod = np.clip(data * (env[:,np.newaxis] if data.ndim>1 else env), -1.0, 1.0).astype(np.float32)
        wavfile.write(out, sr, mod)

        print()
        print(f"  Written to      : {out}")
        print(f"  Sample rate     : {sr} Hz")
        print(f"  Duration        : {data.shape[0]/sr:.1f} s")
        print(f"  Beat frequency  : {beat} Hz")
        print(f"  Modulation depth: {depth*100:.0f}%")
    except Exception as e:
        print(f"  Error processing WAV: {e}"); return

# ---------------------------------------------------------------------------
#  CONSOLE UI
# ---------------------------------------------------------------------------

W = 64

def rule(c=""):
    print(c * W)

def banner():
    print()

def ask_int(prompt, lo, hi):
    while True:
        try:
            v = int(input(prompt).strip())
            if lo <= v <= hi: return v
        except ValueError: pass

def ask_float(prompt):
    while True:
        try:
            v = float(input(prompt).strip())
            if v > 0: return v
        except ValueError: pass

def choose_amplitude():
    global _amplitude
    print()
    rule("-")
    print("  Volume / amplitude  (0.1 = very quiet, 0.8 = default, 1.0 = full)")
    rule("-")
    print()
    print("  Recommended: 0.6-0.8.  Higher levels may cause listener fatigue.")
    print("  Garcia-Argibay (2018): carrier tone volume is not a significant")
    print("  moderator of binaural beat efficacy.")
    v = input(f"\n  Amplitude [0.8]: ").strip()
    try:
        val = float(v) if v else 0.8
        _amplitude = max(0.05, min(1.0, val))
    except ValueError:
        _amplitude = 0.8
    print(f"  Amplitude set to {_amplitude:.2f}")

def choose_pink_noise():
    global _pink_lvl
    print()
    rule("-")
    print("  Pink noise masking (optional)")
    rule("-")
    print()
    print("  Garcia-Argibay et al. (2018) meta-regression: binaural beats masked")
    print("  with pink or white noise show similar efficacy to unmasked beats.")
    print("  Chaieb et al. (2015): pink noise may amplify the binaural beat")
    print("  percept (Oster 1973).  Pink noise can also reduce tinnitus.")
    print()
    print("  0  None  (pure beat tone — default)")
    print("  1  Subtle  (10% pink noise blend)")
    print("  2  Moderate  (25% pink noise blend)")
    print("  3  Heavy  (50% pink noise blend — beat remains audible)")
    print("  4  Custom  (enter exact level 0.0-1.0)")
    print()
    c = ask_int("  Enter number (0-4): ", 0, 4)
    levels = {0: 0.0, 1: 0.10, 2: 0.25, 3: 0.50}
    if c in levels:
        _pink_lvl = levels[c]
    else:
        v = input("  Pink noise level (0.0-1.0): ").strip()
        try:    _pink_lvl = max(0.0, min(1.0, float(v)))
        except: _pink_lvl = 0.0
    if _pink_lvl > 0:
        print(f"  Pink noise blend: {_pink_lvl*100:.0f}%")
    else:
        print("  No pink noise masking.")

def choose_suite():
    print()
    rule("=")
    print("  Select a frequency suite")
    rule("=")
    print()
    print("  1  General Brainwave Suite")
    print("       Standard EEG bands: delta, theta, Schumann resonance, alpha,")
    print("       15 Hz caution, low beta, high beta, gamma.  All descriptions")
    print("       reference peer-reviewed EEG literature.")
    print()
    print("  2  Attention & Cognition Suite")
    print("       Research-specific presets for attention, memory, and ADHD.")
    print("       Engelbregt 2021  |  Malandrone 2022  |  Garcia-Argibay 2017")
    print("       Colzato 2017  |  Kraus & Porubanová 2015")
    print()
    print("  3  MindWar Suite  (Aquino 2016)")
    print("       Frequencies from Michael Aquino's MindWar operational BWR")
    print("       framework.  Includes infrasonic and ELF variants.")
    print()
    print("  4  Alzheimer's / 40 Hz Suite")
    print("       MIT Picower GENUS protocols targeting 40 Hz gamma for")
    print("       amyloid reduction and cognitive preservation.")
    print()
    print("  5  Manual Configuration")
    print("       Specify mode, frequency, carrier and duration directly.")
    print()
    print("  6  Embed Mode")
    print("       AM-modulate an existing WAV file at a target beat frequency.")
    print()
    return ask_int("  Enter number (1-6): ", 1, 6)

def choose_preset(suite, title):
    print()
    rule("=")
    print(f"  {title}")
    rule("=")
    for i,p in enumerate(suite,1):
        print()
        print(f"  {i}  {p['label']}")
        for line in p['desc'].splitlines():
            print(f"  {line}")
    print()
    rule("-")
    return suite[ask_int(f"  Enter number (1-{len(suite)}): ", 1, len(suite))-1]

def choose_duration(p):
    min_s  = MIN.get(p["band"], 0)
    timing = p.get("timing", "during")
    print()
    rule("-")
    print("  Session duration")
    rule("-")
    print()
    if min_s > 0:
        print(f"  Research-recommended minimum for this frequency band: {fmt(min_s)}")
        print()
    print("  1  Continuous  (press Ctrl+C to stop)")
    print("  2  Timed session")
    if timing in ("before", "before_and_during"):
        print("  3  Pre-task induction  (play N min, alert to begin task, then continue)")
        print(f"       Recommended: Garcia-Argibay (2018) found 'before' and")
        print(f"       'before+during' exposure yields larger effects than during-only.")
        hi = 3
    else:
        hi = 2
    c = ask_int(f"\n  Enter number (1-{hi}): ", 1, hi)

    pretask = 0.0
    if c == 1:
        return 0.0, pretask
    if c == 3:
        print()
        print("  Enter pre-task induction duration:")
        h = m = s = 0
        try:
            h = int(input("  Hours   : ").strip() or 0)
            m = int(input("  Minutes : ").strip() or 0)
            s = int(input("  Seconds : ").strip() or 0)
        except ValueError: pass
        pretask = float(h*3600 + m*60 + s)
        if pretask <= 0: pretask = 0.0
        print()
        print("  Now enter total session duration (induction + task):")

    h = m = s = 0
    try:
        h = int(input("  Hours   : ").strip() or 0)
        m = int(input("  Minutes : ").strip() or 0)
        s = int(input("  Seconds : ").strip() or 0)
    except ValueError: pass
    total = h*3600 + m*60 + s
    if total <= 0:
        print("  Zero entered.  Switching to continuous.")
        return 0.0, pretask
    if min_s > 0 and total < min_s:
        print()
        print(f"  Note: {fmt(total)} is below the research-recommended minimum")
        print(f"  of {fmt(min_s)}.  Entrainment may not be achieved.")
    return float(total), pretask

def choose_sr(p):
    if p["mode"] == "parametric":
        print()
        rule("-")
        print("  Sample rate  (parametric mode requires high sample rate)")
        rule("-")
        print()
        print("  Parametric mode generates ultrasonic carriers at 40,000 Hz")
        print("  and 40,040 Hz.  A sample rate of at least 96,000 Hz is needed")
        print("  to represent these frequencies without aliasing.")
        print()
        print("  1  96000 Hz   (recommended)")
        print("  2  192000 Hz  (use if your interface supports it)")
        return {1:96000, 2:192000}[ask_int("\n  Enter number (1-2): ", 1, 2)]
    if p["mode"] == "subliminal":
        print()
        rule("-")
        print("  Sample rate  (subliminal mode works best at 96000 Hz)")
        rule("-")
        print()
        print("  The subliminal carrier is at 18,500 Hz.  At a 44,100 Hz")
        print("  sample rate this sits close to the Nyquist limit and may")
        print("  alias.  96,000 Hz gives a clean carrier.")
        print()
        print("  1  44100 Hz   (standard, acceptable)")
        print("  2  96000 Hz   (recommended)")
        return {1:44100, 2:96000}[ask_int("\n  Enter number (1-2): ", 1, 2)]
    return SR_DEFAULT

def manual_config():
    print()
    rule("=")
    print("  Manual Configuration")
    rule("=")
    print()
    print("  Delivery modes")
    print()
    print("  1  binaural     — two carriers, one per ear; headphones required")
    print("  2  isochronic   — AM-modulated carrier; speakers or headphones")
    print("  3  mono         — direct sine wave; no entrainment mechanism")
    print("  4  infrasonic   — sub-20 Hz pressure wave; subwoofer required")
    print("  5  subliminal   — near-threshold carrier; 96 kHz SR recommended")
    print("  6  parametric   — ultrasonic carriers; 96 kHz + specialist transducer")
    print("  7  surround     — isochronic to all output channels")
    print()
    modes = ["binaural","isochronic","mono","infrasonic","subliminal","parametric","surround"]
    c    = ask_int("  Enter number (1-7): ", 1, 7)
    mode = modes[c-1]

    if mode == "mono":
        hz = ask_float("  Frequency (Hz): ")
        return dict(mode="mono", carrier_hz=hz, beat_hz=0.0, duty=0.5,
                    band="other", label=f"Mono {hz} Hz", desc="",
                    detail="Direct sine wave.", evidence="N/A", timing="during")

    beat = ask_float("  Beat frequency (Hz): ")

    if mode == "infrasonic":
        carrier = beat
    elif mode == "subliminal":
        v = input(f"  Carrier Hz [{SUBLIMINAL_CARRIER}]: ").strip()
        try:    carrier = float(v) if v else SUBLIMINAL_CARRIER
        except: carrier = SUBLIMINAL_CARRIER
    elif mode == "parametric":
        carrier = PARAMETRIC_BASE
    else:
        v = input(f"  Carrier Hz [{CARRIER_OPTIMAL}]: ").strip()
        try:    carrier = float(v) if v else CARRIER_OPTIMAL
        except: carrier = CARRIER_OPTIMAL

    duty = 0.5
    if mode in ("isochronic","surround"):
        v = input("  Duty cycle 0.0-1.0 [0.5]: ").strip()
        try: duty = float(v) if v else 0.5
        except: pass

    return dict(mode=mode, carrier_hz=carrier, beat_hz=beat, duty=duty,
                band="other", label=f"Manual {mode} {beat} Hz",
                desc="", detail="", evidence="N/A", timing="during")

def print_session_header(p, duration, sr, pretask):
    print()
    rule("=")
    print(f"  {p['label']}")
    rule("=")
    print()
    for line in p.get("detail","").splitlines():
        print(f"  {line}")
    print()
    rule("-")
    print(f"  Mode         : {p['mode']}")
    print(f"  Beat target  : {p['beat_hz']} Hz")
    if p["mode"] not in ("infrasonic",):
        print(f"  Carrier      : {p['carrier_hz']} Hz")
    if p["mode"] in ("isochronic","surround"):
        print(f"  Duty cycle   : {p['duty']}")
    print(f"  Sample rate  : {sr} Hz")
    print(f"  Fade         : {FADE_S} s in / {FADE_S} s out")
    print(f"  Volume       : {_amplitude:.2f}")
    if _pink_lvl > 0:
        print(f"  Pink noise   : {_pink_lvl*100:.0f}% blend")
    if pretask > 0:
        print(f"  Pre-task     : {fmt(pretask)} induction, then task-begin alert")
    if duration:
        print(f"  Duration     : {fmt(int(duration))}")
    else:
        print(f"  Duration     : Continuous  (Ctrl+C to stop)")
    min_s = MIN.get(p["band"],0)
    if min_s > 0:
        print(f"  Min session  : {fmt(min_s)}")
    ev = p.get("evidence","")
    if ev:
        print(f"  Evidence     : {ev}")
    timing = p.get("timing","during")
    timing_labels = {
        "during":           "concurrent with task",
        "before":           "induction before task",
        "before_and_during":"before + during task (recommended by Garcia-Argibay 2018)",
    }
    print(f"  Timing       : {timing_labels.get(timing, timing)}")
    rule("-")
    print()
    m = p["mode"]
    if m == "binaural":
        print("  Headphones required.  Binaural does not work on speakers.")
    elif m == "infrasonic":
        print("  Sub-20 Hz capable subwoofer required.")
        print("  Sit or lie in the room -- coupling is through skin and bone.")
    elif m == "subliminal":
        print("  Can be played in the background.  Subject may hear nothing.")
    elif m == "parametric":
        print("  96 kHz interface and transducer passing above 40 kHz required.")
    elif m == "surround":
        print("  Signal will fill all available output channels.")
        print("  All persons in the room will be exposed.")
    else:
        print("  Quiet room recommended.  Eyes closed, passive listening.")
        print("  Do not mix with background music -- this reduces efficacy.")
        print("  Pink noise masking is acceptable (Garcia-Argibay 2018).")

def signal_handler(sig, frame):
    e = time.time()-_t0_wall if _t0_wall else 0
    print(f"\n\n  Stopped.  Session time: {fmt(e)}")
    stop_event.set()

# ---------------------------------------------------------------------------
#  MAIN
# ---------------------------------------------------------------------------

def main():
    global _t0_wall
    signal.signal(signal.SIGINT, signal_handler)
    banner()

    choice = choose_suite()

    if choice == 6:
        run_embed(); return

    if   choice == 1: p = choose_preset(SUITE_GENERAL,    "General Brainwave Suite")
    elif choice == 2: p = choose_preset(SUITE_ATTENTION,  "Attention & Cognition Suite")
    elif choice == 3: p = choose_preset(SUITE_MINDWAR,    "MindWar Suite  (Aquino 2016)")
    elif choice == 4: p = choose_preset(SUITE_ALZHEIMERS, "Alzheimer's / 40 Hz Suite")
    else:             p = manual_config()

    choose_amplitude()
    choose_pink_noise()
    duration, pretask = choose_duration(p)
    sr                = choose_sr(p)
    print_session_header(p, duration, sr, pretask)
    input("  Press Enter to begin.  Ctrl+C to stop.\n")

    t_start = time.time()
    try:
        run_session(p, duration, sr, pretask)
    except Exception as ex:
        print(f"\n  Error: {ex}"); sys.exit(1)

    elapsed = time.time() - t_start
    min_s   = MIN.get(p["band"], 0)
    met_min = elapsed >= min_s if min_s > 0 else True

    print(f"  Session complete.  Total time: {fmt(elapsed)}")
    if min_s > 0:
        if met_min:
            print(f"  Research minimum of {fmt(min_s)} was met.")
        else:
            print(f"  Session ended {fmt(min_s-elapsed)} before the research")
            print(f"  minimum of {fmt(min_s)}.  Entrainment may not have been achieved.")
    print()


if __name__ == "__main__":
    main()
