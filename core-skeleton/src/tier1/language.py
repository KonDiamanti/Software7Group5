# language.py
# is the prompt English? the tier-1 model only knows English,
# so anything else should go to the guard instead of getting a made up score

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0   # same answer every run
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


def is_english(text):
    if not AVAILABLE:
        return True   # cannot check, so we do not block anything because of it
    try:
        return detect(text) == "en"
    except Exception:
        return False   # very short or odd texts can make detect() crash