"""
Requirements
  pip install numpy sounddevice
  pip install scipy   (embed mode only)
"""

import numpy as np
import sounddevice as sd
import threading
import signal
import sys
import queue
import time
import os

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

# Research-backed minimum session lengths in seconds.
# Source: Ingendoh et al. (2023) PLoS ONE systematic review of 14 EEG studies.
MIN = {"delta": 1200, "theta": 360, "alpha": 300, "gamma": 900, "other": 0}

# ---------------------------------------------------------------------------
#  PRESET DATA
#
#  desc   : 2-3 line plain-text summary shown in the selection menu
#  detail : full paragraph shown in the pre-session summary screen
# ---------------------------------------------------------------------------

SUITE_GENERAL = [
    dict(
        label   = "Delta  1.5 Hz  --  Deep sleep / physical recovery",
        mode    = "binaural", beat_hz=1.5, carrier_hz=400.0,
        duty    = 0.5, band="delta",
        desc    = """\
    Dominant during NREM stage 3 deep sleep.  Associated with physical
    restoration, immune regulation and growth hormone release.
    Waking delta does not replicate sleep benefits.  Use for sleep onset.
    Headphones required.  Research minimum: 20 minutes.""",
        detail  = """\
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
    evidence for waking delta entrainment is limited.""",
    ),
    dict(
        label   = "Theta  6 Hz  --  Relaxation / creativity / emotional processing",
        mode    = "binaural", beat_hz=6.0, carrier_hz=400.0,
        duty    = 0.5, band="theta",
        desc    = """\
    Prominent during light sleep, hypnagogic states and deep meditation.
    Associated with memory consolidation, emotional processing and creative
    ideation.  EEG-confirmed at parietal and temporal sites after 6 minutes.
    Headphones required.  Research minimum: 6 minutes.""",
        detail  = """\
    Theta oscillations (4-7 Hz) are prominent during light sleep, the hypnagogic
    (pre-sleep) state and deep meditative states.  They are associated with
    memory consolidation, emotional processing and creative ideation.  Elevated
    waking theta in the frontal midline has been correlated with focused internal
    attention in some research contexts.  However, prolonged waking theta can
    also correlate with inattentiveness and drowsiness.

    EEG confirmation: theta power increase at parietal and temporal electrode
    sites after 6 minutes of stimulation at 400 Hz carrier.  The source study
    (Jirakittayakorn & Wongsawat, 2017, Frontiers in Neuroscience) used 30-minute
    sessions.  400 Hz is the research-optimal carrier for binaural beat perception.

    Note: theta stimulation during cognitively demanding active tasks may reduce
    rather than enhance performance.  Best used during rest or passive states.

    Delivery: binaural beat.  Headphones required.  Research minimum: 6 minutes.""",
    ),
    dict(
        label   = "Schumann  7.83 Hz  --  Earth electromagnetic resonance",
        mode    = "binaural", beat_hz=7.83, carrier_hz=400.0,
        duty    = 0.5, band="theta",
        desc    = """\
    The fundamental resonance frequency of the Earth-ionosphere cavity.
    Falls at the alpha-theta boundary.  Referenced in grounding and
    stress-relief contexts.  Peer-reviewed evidence is limited.
    Best format per BWR Manual: Mono or Binaural.
    Research minimum: 6 minutes.""",
        detail  = """\
    Source: BWR Manual; geophysical literature.

    7.83 Hz is the fundamental resonance frequency of the electromagnetic cavity
    formed between the Earth's surface and the ionosphere, driven by global
    lightning activity.  It coincidentally falls near the alpha-theta boundary
    of human brainwave activity.

    Some researchers hypothesise a biological entrainment relationship between
    the geomagnetic field and human neural oscillations.  Direct peer-reviewed
    evidence of measurable therapeutic or neurological benefit from 7.83 Hz
    audio stimulation in humans remains limited and inconclusive.

    This preset is included as a reference for grounding and stress-relief
    applications.  Claims about Schumann resonance therapy should be treated
    with caution.  Consult primary literature before drawing conclusions.

    Best format per BWR Manual: Mono or Binaural.  This preset uses binaural.
    To use mono delivery instead, select Manual Configuration and choose
    mono mode with 7.83 Hz.

    Delivery: binaural beat.  Headphones required.  Research minimum: 6 minutes.""",
    ),
    dict(
        label   = "Alpha  10 Hz  --  Relaxed wakefulness / stress reduction",
        mode    = "binaural", beat_hz=10.0, carrier_hz=400.0,
        duty    = 0.5, band="alpha",
        desc    = """\
    Dominant during relaxed, eyes-closed wakefulness and light meditation.
    EEG-confirmed increase in frontal, central, parietal and occipital
    alpha power after 5 minutes.  Kim et al. (2023).
    Headphones required.  Research minimum: 5 minutes.""",
        detail  = """\
    Alpha oscillations (8-12 Hz) are the dominant rhythm during relaxed,
    wakeful states such as quiet reflection, closed-eye rest and light
    meditation.  They are associated with reduced cortical arousal, lower
    anxiety and a receptive cognitive state.

    Posterior alpha typically increases when eyes are closed, reflecting
    reduced visual processing load -- a passive reflex, not an actively
    induced benefit.  Alpha suppression (event-related desynchronization)
    occurs naturally during active task engagement and is a sign of normal
    cortical function, not a deficit.

    EEG confirmation: increased absolute alpha power across frontal, central,
    parietal and occipital electrode areas after 5 minutes of stimulation
    at 400 Hz carrier.  Source: Kim et al. (2023), Technology and Health Care.

    Alpha increase is a correlate of reduced arousal rather than a direct
    cause of calm.  Alpha suppression during active task engagement is normal
    and expected -- artificially increasing alpha power does not guarantee
    a shift in subjective state.

    Delivery: binaural beat at 400 Hz carrier.  Headphones required.
    Research minimum: 5 minutes.""",
    ),
    dict(
        label   = "15 Hz  --  Discomfort threshold  (caution)",
        mode    = "mono", beat_hz=15.0, carrier_hz=15.0,
        duty    = 0.5, band="other",
        desc    = """\
    Not recommended for relaxation or general use.  Stimulation near 15 Hz
    has been associated with discomfort, eye strain and headaches in
    sensitive individuals, partly due to proximity to the alpha-beta boundary.
    Auditory stimulation carries a lower but non-zero risk of overstimulation.
    Mono delivery.  Restrict to controlled contexts only.""",
        detail  = """\
    Source: BWR Manual.

    Stimulation near 15 Hz -- particularly via visual flicker -- has been
    associated with discomfort, eye strain and headaches in sensitive
    individuals.  This is partly explained by proximity to the alpha-beta
    boundary and susceptibility in photosensitive subjects.

    Auditory stimulation at this frequency carries a lower but non-zero risk
    of overstimulation.

    The BWR Manual explicitly states: "Not recommended for relaxation or
    general use."  This preset is included for reference completeness only.
    Restrict to controlled contexts with appropriate screening.

    15 Hz is also the lower bound of conscious hearing cited in MindWar
    (Aquino 2016), though modern audiology places this threshold closer to
    20 Hz for most individuals.

    Delivery: mono tone.  Best format per BWR Manual: Mono.
    No therapeutic use recommended.""",
    ),
    dict(
        label   = "Low Beta  16 Hz  --  Focused attention / logical reasoning",
        mode    = "binaural", beat_hz=16.0, carrier_hz=400.0,
        duty    = 0.5, band="other",
        desc    = """\
    Low beta (13-20 Hz) correlated with sustained attention, working memory,
    and deliberate information processing via frontoparietal attention networks.
    Caution: zero of 14 peer-reviewed EEG studies found reliable beta
    entrainment via binaural beats (Ingendoh 2023).  Headphones required.""",
        detail  = """\
    Source: BWR Manual; Ingendoh et al. (2023).

    The beta band (13-30 Hz) is divided into two sub-bands:

    Low beta (13-20 Hz): Correlated with focused attention, logical reasoning,
    and deliberate information processing.  EEG studies consistently observe
    elevated low beta during tasks involving sustained attention, working memory,
    and executive control.  This reflects activity across frontoparietal networks
    including dorsolateral prefrontal and posterior parietal cortices.

    High beta (20-30 Hz): Associated with heightened arousal and cortical
    activation.  In excess or in stress-prone individuals, high beta correlates
    with anxiety, rumination and overstimulation.  High-end beta stimulation
    is not recommended for individuals prone to anxiety or stress sensitivity.

    Mechanistically, beta activity reflects thalamocortical oscillatory loops
    that regulate the signal-to-noise ratio of cortical processing --
    effectively gating which inputs receive attentional priority.
    Neuromodulators such as dopamine and norepinephrine influence this gating
    and contribute to the arousal-dependent variability seen in beta power
    across individuals.

    IMPORTANT LIMITATION: Zero of 14 peer-reviewed EEG studies found reliable
    beta-band entrainment via binaural beats (Ingendoh et al. 2023 systematic
    review, PLoS ONE).  The full beta range extends to 30 Hz; references citing
    22 Hz as the upper boundary (including MindWar) use a non-standard truncation.
    This preset is included for reference and completeness only.

    Delivery: binaural beat.  Headphones required.  No established minimum.""",
    ),
    dict(
        label   = "High Beta  25 Hz  --  Heightened arousal  (caution)",
        mode    = "binaural", beat_hz=25.0, carrier_hz=400.0,
        duty    = 0.5, band="other",
        desc    = """\
    High beta (20-30 Hz) associated with heightened cortical arousal.
    In excess or in stress-prone individuals, correlates with anxiety,
    rumination and overstimulation.  Not recommended for individuals
    prone to anxiety.  Zero EEG studies confirm entrainment at this band.
    Headphones required.  No established research minimum.""",
        detail  = """\
    Source: BWR Manual; Ingendoh et al. (2023).

    High beta oscillations (20-30 Hz) are associated with heightened cortical
    arousal and activation.  In excess or in stress-prone individuals, high beta
    correlates with anxiety, rumination and overstimulation.

    The BWR Manual explicitly states: "High-end beta stimulation is not
    recommended for individuals prone to anxiety or stress sensitivity."

    IMPORTANT LIMITATION: Zero of 14 peer-reviewed EEG studies found reliable
    beta-band entrainment via binaural beats (Ingendoh et al. 2023).  There is
    no empirical support for externally inducing high beta states via audio.

    This preset is included for completeness of the beta band coverage.
    It has no recommended therapeutic application.

    Delivery: binaural beat.  Headphones required.  No established minimum.""",
    ),
    dict(
        label   = "Gamma  40 Hz  --  High cognition / neural coordination",
        mode    = "isochronic", beat_hz=40.0, carrier_hz=200.0,
        duty    = 0.5, band="gamma",
        desc    = """\
    Correlated with high-level information processing, perceptual binding,
    and distributed neural coordination.  BWR Manual recommends Mono or
    Binaural; isochronic used here because 40 Hz exceeds the binaural beat
    perception limit (Perrott & Nelson 1969).  See detail screen.
    Speakers or headphones.  Research minimum: 15 minutes.""",
        detail  = """\
    Source: Jirakittayakorn & Wongsawat (2017b); MIT Picower Institute;
    BWR Manual.

    Gamma oscillations (40-70 Hz) are correlated with high-level information
    processing, cross-regional neural coordination, perceptual binding, working
    memory maintenance and executive function.  Gamma deficits are documented
    in Alzheimer's disease and schizophrenia.

    FORMAT NOTE: The BWR Manual recommends Mono or Binaural for 40 Hz gamma
    (binaural preferred for headphone use; mono for speaker-based environments).
    This preset uses isochronic mode instead, for the following reason:

    40 Hz exceeds the approximately 30 Hz binaural beat perception threshold
    (Perrott & Nelson 1969).  Above this limit the two carrier tones are heard
    separately rather than fusing into a perceived beat.  Isochronic amplitude
    modulation does not rely on this fusion mechanism and is more reliable for
    gamma delivery.  To use mono or binaural delivery as the Manual specifies,
    select Manual Configuration.

    EEG confirmation: gamma power increase in temporal, frontal and central
    regions after 15 minutes.  Source: Jirakittayakorn & Wongsawat (2017b).

    40 Hz is also the foundational frequency in MIT's GENUS Alzheimer's
    research.  See the Alzheimer's suite for full clinical protocol presets.

    Delivery: isochronic tone at 200 Hz carrier.  Speakers or headphones.
    Research minimum: 15 minutes.""",
    ),
]

SUITE_MINDWAR = [
    dict(
        label   = "Delta  2 Hz  --  Deep sleep / cognitive suppression",
        mode    = "binaural", beat_hz=2.0, carrier_hz=400.0,
        duty    = 0.5, band="delta",
        desc    = """\
    MindWar characterisation: "1-3 Hz, characteristic of deep sleep."
    Aquino describes delta BWR as making complex or creative effort
    "exhausting and fruitless."  Consistent with standard neuroscience.
    Headphones required.  Research minimum: 20 minutes.""",
        detail  = """\
    Source: Aquino, MindWar 2nd ed. (2016), Chapter 3, PSYCON #2: Brainwave
    Resonance.

    MindWar characterisation: "1-3 Hz = delta waves, characteristic of deep
    sleep."  Aquino describes the operational use of delta BWR as suppressing
    beta activity in a target population, making anything complex or creative
    "exhausting and fruitless."

    This characterisation is consistent with standard neuroscience.  Delta is
    the dominant oscillatory pattern during NREM stage 3 sleep and is not
    associated with productive waking cognition.

    In Aquino's operational framework, delta BWR is used to degrade the
    cognitive capacity of an adversary.

    Delivery: binaural beat.  Headphones required.
    Research minimum: 20 minutes.  Effect evidence is limited.""",
    ),
    dict(
        label   = "Theta  5.5 Hz  --  Emotional arousal / frustration",
        mode    = "binaural", beat_hz=5.5, carrier_hz=400.0,
        duty    = 0.5, band="theta",
        desc    = """\
    MindWar characterisation: "4-7 Hz, characteristic of high emotion,
    violence and frustration."  This diverges from mainstream research,
    which associates theta with relaxation and meditative states.
    Headphones required.  Research minimum: 6 minutes.""",
        detail  = """\
    Source: Aquino, MindWar 2nd ed. (2016), Chapter 3, PSYCON #2.

    MindWar characterisation: "4-7 Hz = theta waves, characteristic of high
    emotion, violence and frustration."  In Aquino's framework, theta BWR
    is used to promote emotional volatility and impair rational decision-making
    in a target population.

    DIVERGES FROM CONSENSUS: This framing conflicts with mainstream peer-reviewed
    EEG literature, which primarily associates theta with relaxation, memory
    consolidation and meditative states.  Aquino's characterisation appears to
    reflect an older or selectively sourced interpretation of the theta band.

    Delivery: binaural beat.  Headphones required.  Research minimum: 6 minutes.""",
    ),
    dict(
        label   = "Alpha  10 Hz  --  Relaxed / cooperative state",
        mode    = "binaural", beat_hz=10.0, carrier_hz=400.0,
        duty    = 0.5, band="alpha",
        desc    = """\
    MindWar characterisation: "8-12 Hz, characteristic of meditation,
    relaxation and searching for patterns."  Aquino describes alpha BWR
    as enabling "relaxed, pleasant, cooperative discussion."  Consistent
    with mainstream EEG literature.  Headphones required.  Min: 5 minutes.""",
        detail  = """\
    Source: Aquino, MindWar 2nd ed. (2016), Chapter 3, PSYCON #2.

    MindWar characterisation: "8-12 Hz = alpha waves, characteristic of
    meditation, relaxation and searching for patterns."  Aquino describes
    alpha BWR as enabling "relaxed, pleasant, cooperative discussion" -- the
    most overtly positive of his four primary brainwave targets.

    This characterisation is consistent with mainstream EEG literature.
    EEG confirmation: alpha power increase across frontal, central, parietal,
    and occipital areas after 5 minutes (Kim et al. 2023).

    Aquino also uses 10 Hz visual flicker as a "common example of visual
    resonance" in the PSYCON framework, citing the seizures some individuals
    experience when exposed to light flickering at 10 Hz.  This is consistent
    with photosensitive epilepsy literature, which documents sensitivity across
    approximately 10-25 Hz.

    Delivery: binaural beat at 400 Hz carrier.  Headphones required.
    Research minimum: 5 minutes.""",
    ),
    dict(
        label   = "Beta  17 Hz  --  Deliberate effort / logical thought",
        mode    = "mono", beat_hz=17.0, carrier_hz=17.0,
        duty    = 0.5, band="other",
        desc    = """\
    MindWar characterisation: "13-22 Hz, frontal brain activity, deliberate
    effort, logical thought."  Aquino uses a 22 Hz ceiling vs the modern 30 Hz
    standard.  BWR Manual flags 17 Hz for restlessness and anxiety risk.
    Best format per BWR Manual: Mono.  No peer-reviewed EEG entrainment support.""",
        detail  = """\
    Source: Aquino, MindWar 2nd ed. (2016), Chapter 3, PSYCON #2;
    BWR Manual.

    MindWar characterisation: "13-22 Hz = beta waves, characteristic of frontal
    brain activity, deliberate effort and logical thought."  Aquino's upper beta
    ceiling of 22 Hz is narrower than the modern standard of 13-30 Hz.  High beta
    (22-30 Hz) is associated with heightened arousal and anxiety and is absent
    from his framework.

    CAUTION: 17 Hz is flagged in the BWR Manual for restlessness, discomfort,
    and anxiety risk in sensitive individuals.  Visual flicker near 18 Hz has
    been documented to produce visual disturbances in photosensitive subjects.
    These responses are not universal but depend on individual sensitivity.

    The BWR Manual specifies Best Format as Mono for 17 Hz.  Zero of 14 peer-
    reviewed EEG studies confirmed reliable beta entrainment via binaural beats.

    Not recommended for individuals prone to anxiety.
    Delivery: mono tone.  No established research minimum.""",
    ),
    dict(
        label   = "Infrasonic  12.5 Hz  --  Covert subliminal vector (subliminal mode)",
        mode    = "subliminal", beat_hz=12.5, carrier_hz=SUBLIMINAL_CARRIER,
        duty    = 0.5, band="alpha",
        desc    = """\
    MindWar's primary SLIPC delivery vector.  10-15 Hz described as "too low
    to be consciously detected but capable of inducing resonance in the brain."
    This preset AM-modulates an 18,500 Hz carrier at 12.5 Hz.  The carrier
    may be inaudible.  Recommended: 96 kHz sample rate.  Min: 5 minutes.""",
        detail  = """\
    Source: Aquino, MindWar 2nd ed. (2016), Chapter 3, PSYCON #1 and #2.

    MindWar's primary SLIPC (Subliminal Involuntary Psychocontrol) delivery
    vector.  Aquino describes 10-15 Hz as "too low to be consciously detected
    but nonetheless capable of inducing resonance in the brain."  He also cites
    Dr. Robert Becker: hippocampal neural activity reaches maximum at 10-15 cps,
    at the dominant micropulsation frequency of the Earth's geomagnetic field.

    This preset uses subliminal AM mode: the carrier (18,500 Hz) is at or above
    the conscious hearing threshold for most adults.  The 12.5 Hz AM envelope is
    delivered without a perceptible tone.  The subject may hear nothing.  This is
    the closest software-level approximation of the covert delivery vector Aquino
    describes.

    LIMITATION: MindWar's true intended delivery is ELF electromagnetic fields
    transmitted through the environment, penetrating walls and the body's skin
    surface.  Acoustic software cannot replicate this.

    Delivery: AM subliminal.  Speakers or headphones.  96 kHz recommended.
    Research minimum: 5 minutes.""",
    ),
    dict(
        label   = "Infrasonic  12.5 Hz  --  Skin-surface pressure wave (infrasonic mode)",
        mode    = "infrasonic", beat_hz=12.5, carrier_hz=12.5,
        duty    = 0.5, band="alpha",
        desc    = """\
    Outputs a true 12.5 Hz pressure wave with no audible carrier.  The wave
    is felt through skin and bone conduction rather than heard.  Closest
    acoustic equivalent to Aquino's skin-surface coupling mechanism.
    Requires a subwoofer capable of sub-20 Hz output.  Min: 5 minutes.""",
        detail  = """\
    Source: Aquino, MindWar 2nd ed. (2016), Chapter 3, PSYCON #1 and #2.

    This preset outputs a real 12.5 Hz pressure wave with no audible carrier
    tone.  At this frequency the wave is not consciously heard but is felt
    through skin and bone conduction across the entire body surface.  This is
    the closest acoustic equivalent to Aquino's skin-surface delivery mechanism.

    Unlike binaural or isochronic modes, there is no audible carrier.  A person
    in the room may be unaware the signal is active if the speaker system
    reproduces sub-20 Hz without producing audible artifacts.

    HARDWARE REQUIRED: Most consumer speakers physically cannot move sufficient
    air at 12.5 Hz.  A subwoofer rated for sub-20 Hz output is required.

    LIMITATION: Acoustic pressure waves are not electromagnetic fields.  They
    will not penetrate walls.  The skin-permeating, environment-filling quality
    of ELF EM fields requires antenna hardware to replicate.

    Delivery: direct infrasonic output.  Sub-20 Hz subwoofer required.
    Research minimum: 5 minutes.""",
    ),
    dict(
        label   = "ELF  57.5 Hz  --  Project Sanguine / biological hazard range",
        mode    = "isochronic", beat_hz=57.5, carrier_hz=200.0,
        duty    = 0.5, band="other",
        desc    = """\
    Center of the 45-70 Hz range used by the U.S. Navy's Project Sanguine
    ELF submarine transmitter.  Aquino cites Becker: these frequencies
    "alter blood chemistry, blood pressure and brain wave patterns."
    Part of Aquino's 30-100 Hz biological hazard range.
    Speakers or headphones.  No established research minimum.""",
        detail  = """\
    Source: Aquino, MindWar 2nd ed. (2016), Chapter 3, PSYCON #1; Becker via
    Aquino.

    The U.S. Navy's Project Sanguine ELF transmitter (Wisconsin/Michigan,
    subsequently renamed Project Seafarer, then Austere ELF, then Project ELF,
    made operational under the Reagan administration) was designed to broadcast
    at 45-70 Hz to communicate with submerged nuclear submarines.

    Aquino cites Dr. Robert Becker's concern that these frequencies are "close
    enough to the Earth's micropulsations that living things are very sensitive
    to them," and that similar fields had been shown to alter blood chemistry,
    blood pressure and brain wave patterns in animals.

    This range forms part of Aquino's 30-100 Hz ELF biological hazard band.
    MindWar states that ELF radiation in this range "appears to interfere with
    the body's normal biological cycles," with cited effects including chronic
    mild stress, impaired immune response, altered blood chemistry and
    disruption of biocycles.

    This preset is included for completeness of the MindWar frequency map.
    Aquino frames this as a hazard range, not a therapeutic one.

    LIMITATION: True ELF effects require electromagnetic antenna hardware.
    This preset is an isochronic acoustic output only.

    Delivery: isochronic tone.  Speakers or headphones.  No established minimum.""",
    ),
]

SUITE_ALZHEIMERS = [
    dict(
        label   = "MIT GENUS  40 Hz  --  Full 1-hour daily protocol",
        mode    = "isochronic", beat_hz=40.0, carrier_hz=200.0,
        duty    = 0.5, band="gamma",
        desc    = """\
    MIT Picower Institute (Tsai lab) protocol.  40 Hz isochronic tone
    replicating the LED-flicker GENUS method.  Sound alone reduced amyloid
    and tau in Alzheimer's mouse models.  Combined light and sound was most
    effective.  Phase III human trials ongoing (Cognito Therapeutics).
    Speakers or headphones.  Recommended: 60 minutes daily.""",
        detail  = """\
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
    in early-stage Alzheimer's patients.  These findings have been partially
    replicated by independent research groups.  MIT spin-off Cognito
    Therapeutics has advanced this into Phase III clinical trials.

    Human clinical efficacy is not yet confirmed at Phase III level.

    FORMAT NOTE: The BWR Manual recommends Mono or Binaural for 40 Hz.
    This preset uses isochronic because 40 Hz exceeds the ~30 Hz binaural
    beat perception threshold (Perrott & Nelson 1969), making isochronic
    AM modulation more reliable for this frequency.  For mono or binaural
    delivery, use Manual Configuration.

    Delivery: isochronic tone at 200 Hz carrier.  Speakers or headphones.
    Protocol: 1 hour daily.  Research minimum: 15 minutes.""",
    ),
    dict(
        label   = "40 Hz Binaural  --  ASSR via headphones",
        mode    = "binaural", beat_hz=40.0, carrier_hz=400.0,
        duty    = 0.5, band="gamma",
        desc    = """\
    40 Hz binaural beat at research-optimal 400 Hz carrier.  40 Hz exceeds
    the binaural beat perception limit but an auditory steady-state response
    (ASSR) is still measurable in EEG.  Gamma power increase confirmed in
    temporal, frontal and central regions after 15 minutes.
    Headphones required.  Research minimum: 15 minutes.""",
        detail  = """\
    Source: Jirakittayakorn & Wongsawat (2017b); Schwarz & Taylor (2005).

    40 Hz binaural beat at the research-optimal 400 Hz carrier frequency
    (Goodin et al.).

    Note: 40 Hz exceeds the approximately 30 Hz binaural beat perception
    threshold (Perrott & Nelson 1969), meaning the two tones may not fuse
    cleanly into a perceived beat.  However, an auditory steady-state response
    (ASSR) is still measurable in EEG at the difference frequency, confirming
    neural processing at 40 Hz.  Gamma power increase has been confirmed in
    temporal, frontal and central regions after 15 minutes.

    Isochronic mode provides more reliable 40 Hz delivery.  This binaural
    variant is an alternative for those who prefer the binaural mechanism or
    are already using headphones.

    Delivery: binaural beat.  Headphones required.
    Research minimum: 15 minutes.""",
    ),
    dict(
        label   = "40 Hz Daily Maintenance  --  Short session (15 min)",
        mode    = "isochronic", beat_hz=40.0, carrier_hz=200.0,
        duty    = 0.5, band="gamma",
        desc    = """\
    Short daily session preset for regular use within a sustained protocol.
    Ismail et al. (2018) found 10-day human trials insufficient for amyloid
    reduction, indicating weeks or months of consistent daily sessions are
    likely required for measurable effect.
    Speakers or headphones.  Research minimum: 15 minutes.""",
        detail  = """\
    Source: Tsai Lab; Ismail et al. (2018), International Journal of
    Alzheimer's Disease.

    Ismail et al. (2018) conducted a human pilot study of 40 Hz LED light
    therapy (2 hours/day, 10 days, 6 amyloid-positive AD/MCI patients with
    PiB PET scans pre and post).  No significant decrease in cortical amyloid
    load was detected.  The authors concluded that 10 days was insufficient
    and that longer-duration therapies are required.

    This preset is designed for daily short sessions as part of a sustained
    longer-term protocol rather than as a standalone treatment.  It is not a
    substitute for the full MIT 1-hour daily protocol.

    Delivery: isochronic tone at 200 Hz carrier.  Speakers or headphones.
    Research minimum: 15 minutes.""",
    ),
    dict(
        label   = "40 Hz Parametric  --  Ultrasonic in-air beat",
        mode    = "parametric", beat_hz=40.0, carrier_hz=PARAMETRIC_BASE,
        duty    = 0.5, band="gamma",
        desc    = """\
    Two ultrasonic carriers at 40,000 Hz and 40,040 Hz.  Neither is audible.
    Their interaction produces a 40 Hz AM envelope in the air itself via
    nonlinear acoustic demodulation.  The beat frequency is generated from
    carriers entirely outside the human hearing range.
    Requires 96 kHz interface and a transducer passing above 40 kHz.""",
        detail  = """\
    Source: Parametric audio principle applied to MIT Picower 40 Hz target.

    Two ultrasonic carrier tones are generated at 40,000 Hz and 40,040 Hz.
    Neither frequency is audible to humans.  Their amplitude-modulated sum
    creates a 40 Hz envelope in the ultrasonic band.  Nonlinear acoustic
    interaction in the air column (the same mechanism used in LRAD directional
    audio systems) demodulates this into a real 40 Hz pressure variation.

    The 40 Hz pressure wave is generated entirely from carriers outside the
    conscious hearing range.  This is the closest software-level analog to a
    delivery mechanism that does not rely on an audible signal.

    HARDWARE REQUIREMENT: Most consumer audio interfaces and DACs roll off
    well before 40 kHz.  A professional interface with flat response to 40+
    kHz and a compatible transducer are required for this mode to function as
    described.  On standard consumer hardware the ultrasonic output will be
    attenuated or absent.

    Delivery: parametric ultrasonic.  96 kHz SR required.  Min: 15 minutes.""",
    ),
    dict(
        label   = "Upper Gamma  60 Hz  --  High cognition / distributed coordination",
        mode    = "isochronic", beat_hz=60.0, carrier_hz=200.0,
        duty    = 0.5, band="gamma",
        desc    = """\
    Upper gamma range (40-70 Hz), correlated with perceptual binding,
    distributed cortical coordination and intense attentional states.
    Gamma deficits are documented in Alzheimer's and schizophrenia.
    Auditory gamma entrainment research is less established than 40 Hz.
    Speakers or headphones.  Research minimum: 15 minutes.""",
        detail  = """\
    Source: BWR Manual; gamma band EEG literature.

    60 Hz sits within the 40-70 Hz gamma range correlated with high-level
    information processing, perceptual binding, cross-regional neural
    coordination and states of intense attentional focus and complex
    cognitive engagement.  Gamma deficits have been documented in Alzheimer's
    disease and schizophrenia.

    Gamma activity is understood as an emergent property of network-level
    neural coordination rather than a discrete frequency that can be reliably
    switched on via external stimulation.

    FORMAT NOTE: The BWR Manual recommends Mono or Binaural for the 40-70 Hz
    gamma range (binaural preferred for headphone use; mono for speaker-based
    environments).  This preset uses isochronic, which is not limited by the
    ~30 Hz binaural perception ceiling and gives more reliable amplitude
    modulation at this frequency.  For mono or binaural delivery as the Manual
    specifies, use Manual Configuration.

    Auditory gamma entrainment research is considerably less established than
    visual (LED flicker) GENUS research.  The 40 Hz target has by far the
    strongest evidence base.  Individual entrainment responses at 60 Hz are
    highly variable.

    Delivery: isochronic tone at 200 Hz carrier.  Speakers or headphones.
    Research minimum: 15 minutes.""",
    ),
]

# ---------------------------------------------------------------------------
#  GLOBAL STATE
# ---------------------------------------------------------------------------

stop_event = threading.Event()
_t0_wall   = None

# ---------------------------------------------------------------------------
#  SIGNAL GENERATORS  (vectorised -- no per-sample loops)
# ---------------------------------------------------------------------------

def _env(t0, n, fi, fo, tot):
    e = np.ones(n, dtype=np.float32)
    if fi > 0 and t0 < fi:
        hi = min(t0+n, fi); k = hi-t0
        e[:k] = np.linspace(t0/fi, hi/fi, k, endpoint=False, dtype=np.float32)
    if tot and fo > 0:
        fs = tot-fo
        if t0+n > fs:
            lb = max(0, fs-t0)
            a0 = max(0.0, (tot-(t0+lb))/fo)
            a1 = max(0.0, (tot-(t0+n))/fo)
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

def stereo(m):   return np.column_stack((m, m))

def gen_binaural(carr, beat, t0, n, sr, fi, fo, tot):
    t=_t(t0,n,sr); e=_env(t0,n,fi,fo,tot)
    L=(np.sin(2*np.pi*carr*t)*e).astype(np.float32)
    R=(np.sin(2*np.pi*(carr+beat)*t)*e).astype(np.float32)
    return np.column_stack((L,R))

def gen_isochronic(carr, beat, t0, n, sr, duty, fi, fo, tot):
    t=_t(t0,n,sr); e=_env(t0,n,fi,fo,tot)
    m=(np.sin(2*np.pi*carr*t)*_gate(t,beat,duty,sr,n)*e).astype(np.float32)
    return stereo(m)

def gen_mono(carr, t0, n, sr, fi, fo, tot):
    t=_t(t0,n,sr); e=_env(t0,n,fi,fo,tot)
    return stereo((np.sin(2*np.pi*carr*t)*e).astype(np.float32))

def gen_infrasonic(beat, t0, n, sr, fi, fo, tot):
    return gen_mono(beat, t0, n, sr, fi, fo, tot)

def gen_subliminal(carr, beat, t0, n, sr, fi, fo, tot):
    t=_t(t0,n,sr); e=_env(t0,n,fi,fo,tot)
    am=(1.0+np.cos(2*np.pi*beat*t))/2.0
    return stereo((np.sin(2*np.pi*carr*t)*am*e).astype(np.float32))

def gen_parametric(base, beat, t0, n, sr, fi, fo, tot):
    t=_t(t0,n,sr); e=_env(t0,n,fi,fo,tot)
    w=((np.sin(2*np.pi*base*t)+np.sin(2*np.pi*(base+beat)*t))*0.5*e).astype(np.float32)
    return stereo(w)

def gen_surround(carr, beat, t0, n, sr, nch, duty, fi, fo, tot):
    blk=gen_isochronic(carr, beat, t0, n, sr, duty, fi, fo, tot)
    return np.tile(blk[:,0:1], (1, nch))

def norm(b):
    p=np.max(np.abs(b)); return b/p if p>1.0 else b

# ---------------------------------------------------------------------------
#  PRODUCER + PLAYBACK
# ---------------------------------------------------------------------------

def producer(p, q, stop_ev, sr, total=None, nch=2):
    fi=int(FADE_S*sr); t0=0
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
        blk=norm(blk)
        if fill < BLOCK_SIZE:
            blk=np.vstack((blk,np.zeros((BLOCK_SIZE-fill,blk.shape[1]),dtype=np.float32)))
        q.put(blk); t0+=fill

def make_cb(q):
    def cb(out,frames,ti,st):
        try:   blk=q.get_nowait()
        except queue.Empty: out[:]=0; return
        if blk is None: raise sd.CallbackStop()
        out[:]=blk.reshape(out.shape)
    return cb

def prefill(q, n=PREFILL//2):
    while q.qsize() < n: pass

def get_nch():
    try:
        d=sd.query_devices(sd.default.device[1],'output')
        return max(2,int(d['max_output_channels']))
    except: return 2

def fmt(s):
    s=int(s); h,r=divmod(s,3600); m,sc=divmod(r,60)
    return f"{h:02d}:{m:02d}:{sc:02d}"

def progress_thread(min_s, total_s):
    global _t0_wall
    _t0_wall=time.time()
    while not stop_event.is_set():
        e  =time.time()-_t0_wall
        pct=min(e/min_s*100,100) if min_s>0 else 100
        bar="="*int(pct/5)+"-"*(20-int(pct/5))
        rem=f"  remaining {fmt(max(0,total_s-e))}" if total_s else "  continuous"
        sys.stdout.write(f"\r  {fmt(e)}  [{bar}] {pct:5.1f}%{rem}   ")
        sys.stdout.flush(); time.sleep(1)
    sys.stdout.write("\n")

def run_session(p, duration, sr):
    nch  =get_nch() if p["mode"]=="surround" else 2
    total=None if not duration else int(duration*sr)
    q    =queue.Queue(maxsize=PREFILL*2)
    local=threading.Event()
    sev  =local if total else stop_event

    thr=threading.Thread(target=producer,args=(p,q,sev,sr,total,nch),daemon=True)
    thr.start(); prefill(q)
    threading.Thread(target=progress_thread,
                     args=(MIN.get(p["band"],0),duration if duration else None),
                     daemon=True).start()

    ch=nch if p["mode"]=="surround" else 2
    with sd.OutputStream(samplerate=sr,channels=ch,
                         blocksize=BLOCK_SIZE,dtype=DTYPE,callback=make_cb(q)):
        if total: sd.sleep(int(duration*1000)); local.set()
        else:     stop_event.wait()
    thr.join(timeout=2)
    if total: stop_event.set()

# ---------------------------------------------------------------------------
#  EMBED MODE
# ---------------------------------------------------------------------------

def run_embed():
    W=64
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
    print("  This is the software implementation of Aquino's 'insertion")
    print("  into electronic media' delivery vector from MindWar.")
    print()
    inp=input("  Input WAV path  : ").strip().strip('"')
    if not os.path.isfile(inp):
        print("  Error: file not found."); return
    out=input("  Output WAV path : ").strip().strip('"')
    while True:
        try:
            beat=float(input("  Beat frequency (Hz) : ").strip())
            if beat>0: break
        except ValueError: pass
    val=input("  Modulation depth 0.0-1.0  [0.15 = subtle / 1.0 = full] : ").strip()
    try:    depth=max(0.0,min(1.0,float(val) if val else 0.15))
    except: depth=0.15

    try:
        from scipy.io import wavfile
    except ImportError:
        print("  Error: scipy not installed.  Run: pip install scipy"); return

    sr,data=wavfile.read(inp)
    orig=data.dtype
    if np.issubdtype(orig,np.integer):
        data=data.astype(np.float32)/np.iinfo(orig).max
    else:
        data=data.astype(np.float32)

    t  =np.arange(data.shape[0],dtype=np.float64)/sr
    env=1.0+depth*np.sin(2*np.pi*beat*t)
    mod=np.clip(data*(env[:,np.newaxis] if data.ndim>1 else env),-1.0,1.0).astype(np.float32)
    wavfile.write(out,sr,mod)

    print()
    print(f"  Written to      : {out}")
    print(f"  Sample rate     : {sr} Hz")
    print(f"  Duration        : {data.shape[0]/sr:.1f} s")
    print(f"  Beat frequency  : {beat} Hz")
    print(f"  Modulation depth: {depth*100:.0f}%")

# ---------------------------------------------------------------------------
#  CONSOLE UI
# ---------------------------------------------------------------------------

W=64

def rule(c=""): print(c*W)

def banner():
    rule()

    rule()

def ask_int(prompt, lo, hi):
    while True:
        try:
            v=int(input(prompt).strip())
            if lo<=v<=hi: return v
        except ValueError: pass

def ask_float(prompt):
    while True:
        try:
            v=float(input(prompt).strip())
            if v>0: return v
        except ValueError: pass

def choose_suite():
    print()
    rule("=")
    print("  Select a frequency suite")
    rule("=")
    print()
    print("  1  General Brainwave Suite")
    print("       Standard EEG bands: delta, theta, Schumann resonance, alpha,")
    print("       15 Hz caution, low beta, high beta, gamma.  All descriptions")
    print("       reference peer-reviewed EEG literature and the BWR Manual.")
    print("       Includes caution entries for 15 Hz and high beta.")
    print()
    print("  2  MindWar Suite  (Aquino 2016)")
    print("       Frequencies from Michael Aquino's MindWar operational BWR")
    print("       framework.  Each entry shows Aquino's characterisation and")
    print("       the scientific consensus where they diverge.  Includes the")
    print("       infrasonic SLIPC vector and ELF / Project Sanguine range.")
    print()
    print("  3  Alzheimer's / 40 Hz Suite")
    print("       MIT Picower GENUS protocols targeting 40 Hz gamma for")
    print("       amyloid reduction and cognitive preservation.  Includes")
    print("       the full 1-hour protocol, daily maintenance and a")
    print("       parametric ultrasonic variant.")
    print()
    print("  4  Manual Configuration")
    print("       Specify mode, frequency, carrier and duration directly.")
    print()
    print("  5  Embed Mode")
    print("       AM-modulate an existing WAV file at a target beat frequency.")
    print("       Implements Aquino's 'insertion into broadcast media' vector.")
    print()
    return ask_int("  Enter number (1-5): ", 1, 5)

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
    min_s=MIN.get(p["band"],0)
    print()
    rule("-")
    print("  Session duration")
    rule("-")
    print()
    if min_s>0:
        print(f"  Research-recommended minimum for this frequency band: {fmt(min_s)}")
        print()
    print("  1  Continuous  (press Ctrl+C to stop)")
    print("  2  Timed session")
    c=ask_int("\n  Enter number (1-2): ",1,2)
    if c==1:
        return 0.0
    h=m=s=0
    try:
        h=int(input("  Hours   : ").strip() or 0)
        m=int(input("  Minutes : ").strip() or 0)
        s=int(input("  Seconds : ").strip() or 0)
    except ValueError: pass
    total=h*3600+m*60+s
    if total<=0:
        print("  Zero entered.  Switching to continuous.")
        return 0.0
    if min_s>0 and total<min_s:
        print()
        print(f"  Note: {fmt(total)} is below the research-recommended minimum")
        print(f"  of {fmt(min_s)}.  Entrainment may not be achieved.")
    return float(total)

def choose_sr(p):
    if p["mode"]=="parametric":
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
        return {1:96000,2:192000}[ask_int("\n  Enter number (1-2): ",1,2)]
    if p["mode"]=="subliminal":
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
        return {1:44100,2:96000}[ask_int("\n  Enter number (1-2): ",1,2)]
    return SR_DEFAULT

def manual_config():
    print()
    rule("=")
    print("  Manual Configuration")
    rule("=")
    print()
    print("  Delivery modes")
    print()
    print("  1  binaural")
    print("       Two carrier tones, one per ear, create a perceived beat at")
    print("       their frequency difference.  Headphones required.  Works")
    print("       reliably up to approximately 30 Hz (Perrott & Nelson 1969).")
    print("       Research-optimal carrier: 400 Hz (Goodin et al.).")
    print()
    print("  2  isochronic")
    print("       A carrier tone is amplitude-modulated at the beat frequency.")
    print("       Works on speakers or headphones.  Preferred for gamma (40 Hz+)")
    print("       because it is not limited by the binaural perception ceiling.")
    print()
    print("  3  mono")
    print("       Direct sine wave at the specified frequency.  No entrainment")
    print("       mechanism.  Use for frequency reference or direct delivery.")
    print()
    print("  4  infrasonic")
    print("       Outputs a true sub-20 Hz pressure wave with no audible carrier.")
    print("       The wave is felt through skin and bone conduction.  Requires a")
    print("       subwoofer capable of sub-20 Hz output.")
    print()
    print("  5  subliminal")
    print("       AM-modulates a near-threshold carrier (default 18,500 Hz) at")
    print("       the beat frequency.  Carrier may be inaudible.  96 kHz SR")
    print("       recommended.")
    print()
    print("  6  parametric")
    print("       Two ultrasonic carriers produce the beat frequency via nonlinear")
    print("       air interaction.  Neither carrier is audible.  Requires a 96 kHz")
    print("       interface and a transducer passing above 40 kHz.")
    print()
    print("  7  surround")
    print("       Isochronic signal broadcast to all available output channels.")
    print("       Room-filling delivery.")
    print()
    modes=["binaural","isochronic","mono","infrasonic","subliminal","parametric","surround"]
    c=ask_int("  Enter number (1-7): ",1,7)
    mode=modes[c-1]

    if mode=="mono":
        hz=ask_float("  Frequency (Hz): ")
        return dict(mode="mono",carrier_hz=hz,beat_hz=0.0,duty=0.5,
                    band="other",label=f"Mono {hz} Hz",desc="",
                    detail="Direct sine wave.",source="manual")

    beat=ask_float("  Beat frequency (Hz): ")

    if mode=="infrasonic":
        carrier=beat
    elif mode=="subliminal":
        v=input(f"  Carrier Hz [{SUBLIMINAL_CARRIER}]: ").strip()
        try:    carrier=float(v) if v else SUBLIMINAL_CARRIER
        except: carrier=SUBLIMINAL_CARRIER
    elif mode=="parametric":
        carrier=PARAMETRIC_BASE
    else:
        v=input(f"  Carrier Hz [{CARRIER_OPTIMAL}]: ").strip()
        try:    carrier=float(v) if v else CARRIER_OPTIMAL
        except: carrier=CARRIER_OPTIMAL

    duty=0.5
    if mode in ("isochronic","surround"):
        v=input("  Duty cycle 0.0-1.0 [0.5]: ").strip()
        try: duty=float(v) if v else 0.5
        except: pass

    return dict(mode=mode,carrier_hz=carrier,beat_hz=beat,duty=duty,
                band="other",label=f"Manual {mode} {beat} Hz",desc="",
                detail="",source="manual")

def print_session_header(p, duration, sr):
    print()
    rule("=")
    print(f"  {p['label']}")
    rule("=")
    print()
    # Print the detail block
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
    if duration:
        print(f"  Duration     : {fmt(int(duration))}")
    else:
        print(f"  Duration     : Continuous  (Ctrl+C to stop)")
    min_s=MIN.get(p["band"],0)
    if min_s>0:
        print(f"  Min session  : {fmt(min_s)}")
    rule("-")
    print()
    m=p["mode"]
    if m=="binaural":
        print("  Headphones required.  Binaural does not work on speakers.")
    elif m=="infrasonic":
        print("  Sub-20 Hz capable subwoofer required.")
        print("  Sit or lie in the room -- coupling is through skin and bone.")
    elif m=="subliminal":
        print("  Can be played in the background.  Subject may hear nothing.")
    elif m=="parametric":
        print("  96 kHz interface and transducer passing above 40 kHz required.")
    elif m=="surround":
        print("  Signal will fill all available output channels.")
        print("  All persons in the room will be exposed.")
    else:
        print("  Quiet room recommended.  Eyes closed, passive listening.")
        print("  Do not mix with background music or noise -- this reduces")
        print("  or eliminates the entrainment effect.")

def signal_handler(sig, frame):
    e=time.time()-_t0_wall if _t0_wall else 0
    print(f"\n\n  Stopped.  Session time: {fmt(e)}")
    stop_event.set()

# ---------------------------------------------------------------------------
#  MAIN
# ---------------------------------------------------------------------------

def main():
    global _t0_wall
    signal.signal(signal.SIGINT, signal_handler)
    banner()

    choice=choose_suite()
    if choice==5:
        run_embed(); return

    if   choice==1: p=choose_preset(SUITE_GENERAL,   "General Brainwave Suite")
    elif choice==2: p=choose_preset(SUITE_MINDWAR,   "MindWar Suite  (Aquino 2016)")
    elif choice==3: p=choose_preset(SUITE_ALZHEIMERS,"Alzheimer's / 40 Hz Suite")
    else:           p=manual_config()

    duration=choose_duration(p)
    sr      =choose_sr(p)
    print_session_header(p, duration, sr)
    input("  Press Enter to begin.  Ctrl+C to stop.\n")

    try:
        run_session(p, duration, sr)
    except Exception as ex:
        print(f"\n  Error: {ex}"); sys.exit(1)

    elapsed=time.time()-_t0_wall if _t0_wall else 0
    min_s  =MIN.get(p["band"],0)
    print(f"  Session complete.  Total time: {fmt(elapsed)}")
    if min_s>0:
        if elapsed>=min_s:
            print(f"  Research minimum of {fmt(min_s)} was met.")
        else:
            print(f"  Session ended {fmt(min_s-elapsed)} before the research")
            print(f"  minimum of {fmt(min_s)}.  Entrainment may not have been achieved.")
    print()

if __name__=="__main__":
    main()
