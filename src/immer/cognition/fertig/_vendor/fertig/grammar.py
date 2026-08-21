"""
fertig.grammar — die strukturelle Schicht: Kongruenz-Regeln.

BLiMP zeigte: n-gram sieht Grammatikalität nicht (26.3%, unter Chance).
Diese Schicht fügt explizite, deterministische Kongruenz-Regeln hinzu:

  R1 Anapher-Genus   : himself/herself  vs. Antezedens-Pronomen
  R2 Anapher-Numerus : themselves/itself vs. Antezedens-Numerus
  R3 Determiner-Nomen: this/that + Plural, these/those + Singular
  R4 Subjekt-Verb    : Plural-Subjekt + 3sg-Verb (und umgekehrt)

Jede Regel liefert +1 (erfüllt), −1 (verletzt), 0 (nicht anwendbar).
Design: Struktur zuerst, Statistik als Backoff — dieselbe Trennung wie
überall in FERTIG (Fakten exakt, Form gemessen).
"""

from __future__ import annotations

import re
from typing import List, Tuple

from .pipeline import _toks

# ---------------------------------------------------------------------------
# Lexika
# ---------------------------------------------------------------------------

_MALE = {"he", "him", "his", "himself", "boy", "man", "men"}
_FEMALE = {"she", "her", "hers", "herself", "girl", "woman", "women"}
_NEUTER = {"it", "its", "itself"}

_FEMALE_NAMES = {"katherine", "mary", "susan", "linda", "jane", "sarah",
                 "anna", "maria", "emma", "olivia", "sophia", "ava",
                 "isabella", "mia", "charlotte", "amelia", "harper",
                 "evelyn", "abigail", "emily", "elizabeth", "samantha",
                 "victoria", "madison", "grace", "chloe", "penelope",
                 "layla", "riley", "zoey", "nora", "lily", "eleanor",
                 "hannah", "lillian", "addison", "aubrey", "ella",
                 "natalie", "camila", "lena", "diana", "julia", "amy",
                 "ann", "rose", "joan", "judy", "martha", "marie"}
_MALE_NAMES = {"john", "james", "robert", "michael", "william", "david",
               "richard", "joseph", "thomas", "charles", "christopher",
               "daniel", "matthew", "anthony", "mark", "donald",
               "steven", "paul", "andrew", "joshua", "kenneth", "kevin",
               "brian", "george", "timothy", "ronald", "edward", "jason",
               "jeffrey", "ryan", "jacob", "gary", "nicholas", "eric",
               "jonathan", "stephen", "larry", "justin", "scott",
               "brandon", "benjamin", "samuel", "gregory", "alexander",
               "frank", "patrick", "raymond", "jack", "dennis", "jerry",
               "tyler", "aaron", "jose", "adam", "nathan", "henry",
               "douglas", "zachary", "peter", "kyle", "walter", "ethan",
               "jeremy", "harold", "keith", "christian", "roger", "noah",
               "gerald", "carl", "terry", "sean", "austin", "arthur",
               "lawrence", "jesse", "dylan", "bryan", "joe", "jordan",
               "billy", "bruce", "albert", "willie", "gabriel", "logan",
               "alan", "juan", "wayne", "roy", "ralph", "randy",
               "eugene", "vincent", "russell", "elijah", "louis",
               "bobby", "philip", "johnny"}
_FEMALE_ROLES = {"actress", "queen", "mother", "sister", "daughter",
                 "aunt", "grandmother", "girl", "woman", "bride",
                 "waitress", "princess", "heroine", "hostess", "widow",
                 "witch", "goddess", "countess", "duchess", "baroness",
                 "empress", "lady"}
_MALE_ROLES = {"actor", "king", "father", "brother", "son", "uncle",
               "grandfather", "boy", "man", "groom", "waiter", "prince",
               "hero", "host", "widower", "wizard", "god", "count",
               "duke", "baron", "emperor", "gentleman"}

_MALE_SIGNALS = _MALE | _MALE_NAMES | _MALE_ROLES
_FEMALE_SIGNALS = _FEMALE | _FEMALE_NAMES | _FEMALE_ROLES

_SINGULAR_PRON = _MALE | _FEMALE | _NEUTER | {"i", "me", "my", "myself",
                                             "you", "your", "yourself"}
_PLURAL_PRON = {"they", "them", "their", "themselves", "we", "us", "our",
                "ourselves", "yourselves"}

_SINGULAR_IRREG = {"child", "person", "man", "woman", "mouse", "foot",
                   "tooth", "goose", "ox", "cactus", "criterion", "datum",
                   "analysis", "basis", "crisis", "thesis", "apparatus"}
_PLURAL_IRREG = {"children", "people", "men", "women", "mice", "feet",
                 "teeth", "geese", "oxen", "cacti", "criteria", "data",
                 "analyses", "bases", "crises", "theses", "apparatuses"}

# 3sg-Verben (endet auf s, mit Ausnahmen) + unregelmäßige 3sg-Formen
_3SG_IRREG = {"is", "has", "does", "was"}
_3SG_SUFFIX_EXCL = ("ss", "is", "us", "ous", "as", "es")  # es: verben wie
# "dresses" sind 3sg, aber "dresses" (Nomen) auch — für v1: nur "es" bei
# Verben mit s-Stamm wie passes/dresses wird über das Nomen-Verb-Muster
# entschieden; hier zählen wir es als 3sg-Signal.


def _is_plural(word: str) -> bool:
    if word in _PLURAL_IRREG:
        return True
    if word in _SINGULAR_IRREG:
        return False
    return word.endswith("s") and not word.endswith(("ss", "is", "us", "ous"))


def _is_3sg(word: str) -> bool:
    if word in _3SG_IRREG:
        return True
    return word.endswith("s") and not word.endswith(("ss", "is", "us", "ous"))


# ---------------------------------------------------------------------------
# Regeln
# ---------------------------------------------------------------------------

# Geschlossene Workklassen: können keine Namen sein (Suffix-Heuristik-
# Schutz) — Auxiliare + Kontraktions-Reste aus _toks
_CLOSED_CLASS = {"hasn", "havent", "doesnt", "didnt", "wont", "cant",
                 "couldnt", "wouldnt", "shouldnt", "isnt", "wasnt",
                 "werent", "dont", "hadnt", "t", "n", "ve", "re", "ll",
                 "m", "s", "d", "has", "have", "had", "do", "does",
                 "did", "can", "could", "will", "would", "should", "may",
                 "might", "must", "is", "are", "was", "were", "be",
                 "been", "being", "not", "no", "the", "a", "an"}


def _antecedent_gender(toks: List[str], up_to: int) -> Optional[str]:
    """Genus-Signal des Antezedens vor Position up_to: 'male'/'female'/None."""
    for prev in toks[:up_to]:
        if prev in _FEMALE_SIGNALS:
            return "female"
        if prev in _MALE_SIGNALS:
            return "male"
        # Suffix-Heuristik NUR für original-großgeschriebene Namen:
        # -a/-ia/-ina ist ein starkes Femininum-Signal, Konsonant maskulin
        if (len(prev) > 3 and prev not in _CLOSED_CLASS
                and prev in _raw_caps):
            if prev.endswith(("a", "ia", "ina")):
                return "female"
            if prev[-1] not in "aeiouy" and not prev.endswith("e"):
                return "male"
    return None


def rule_anaphor_gender(toks: List[str]) -> int:
    """R1 (Principle A): himself/herself bindet an das Hauptsatz-Subjekt.
    Namen in Relativsätzen ('Gregory') sind keine Antezedens-Kandidaten."""
    for i, t in enumerate(toks):
        if t not in ("himself", "herself"):
            continue
        subj = _anaphor_subject(toks, i)
        if subj is None:
            return 0
        g = _antecedent_gender([subj], 1)
        if t == "himself":
            if g == "female":
                return -1
            return 1 if g == "male" else 0
        if g == "male":
            return -1
        return 1 if g == "female" else 0
    return 0


def _preceding_name(toks: List[str], up_to: int) -> bool:
    """Gibt es ein namen-ähnliches Token (unbekannt, kein geschlossenes
    Wort, keine Pronomen, KEIN Plural-Nomen) vor Position up_to?
    Namen sind singular — 'patients' ist plural und kein Name."""
    for prev in toks[:up_to]:
        if prev in (_SINGULAR_PRON | _PLURAL_PRON | _CLOSED_CLASS):
            continue
        if _is_plural(prev) or prev in _PLURAL_IRREG:
            continue  # Plural-Nomen sind keine Singular-Namen
        if len(prev) > 3 and prev in _raw_caps:
            return True  # nur original-großgeschrieben = Name
    return False


_SKIP_SUBJ = _CLOSED_CLASS | {"the", "a", "an", "of", "lot", "lots",
                                    "some", "many", "all", "both", "each",
                                    "every", "this", "that", "these", "those",
                                    "no", "one", "two", "three", "four",
                                    "five", "six", "seven", "eight", "nine",
                                    "ten", "in", "on", "at", "to", "for",
                                    "with", "by", "from", "about", "into",
                                    "through", "toward", "upon", "of"}


_AUX_WORDS = {"is", "are", "was", "were", "has", "have", "had", "did",
               "does", "do", "can", "could", "would", "should", "will",
               "might", "may", "must", "be", "been", "being", "didn",
               "doesn", "isn", "aren", "wasn", "weren", "hasn", "havent",
               "not", "t"}


def _main_subject(toks: List[str]) -> Optional[str]:
    """Hauptsatz-Subjekt (Principle A): das erste nominale Token VOR dem
    letzten Relativpronomen — Namen in Relativsätzen ('Gregory') c-commanden
    das Reflexiv nicht und sind keine Antezedens-Kandidaten. Vorwärts-Scan:
    Englisch ist SVO, das Subjekt steht am Satzanfang (rückwärts trifft
    Verben wie 'hurt')."""
    rel_pos = -1
    for i, t in enumerate(toks):
        if t in ("who", "that", "which", "whom"):
            rel_pos = i
    limit = rel_pos if rel_pos >= 0 else len(toks)
    for i in range(0, limit):
        t = toks[i]
        if t in _SKIP_SUBJ or t in _AUX_WORDS:
            continue
        return t
    # Fallback: rückwärts, nur nominale Kandidaten
    for i in range(limit - 1, -1, -1):
        t = toks[i]
        if t in _SKIP_SUBJ or t in _AUX_WORDS:
            continue
        if t in _PARTICIPLES or t in _PAST_IRREG:
            continue
        return t
    return None


def rule_anaphor_number(toks: List[str]) -> int:
    """R2 (Principle A): Reflexiv bindet an das HAUPTsatz-Subjekt.
    themselves braucht ein Plural-Subjekt, itself ein Singular-Subjekt.
    Namen in Relativsätzen zählen nicht ('children ... hated himself')."""
    for i, t in enumerate(toks):
        if t not in ("themselves", "itself"):
            continue
        subj = _anaphor_subject(toks, i)
        if subj is None:
            continue
        plural = _is_plural(subj) or subj in _PLURAL_IRREG
        singular = (not plural) and (subj in _SINGULAR_IRREG or
                                     subj in _raw_caps or
                                     subj in _SINGULAR_PRON)
        if t == "themselves":
            return 1 if plural else (-1 if singular else 0)
        if t == "itself":
            return 1 if singular else (-1 if plural else 0)
    return 0


_ADJECTIVES = {"big", "small", "old", "new", "young", "red", "blue",
               "green", "black", "white", "tall", "short", "long",
               "large", "little", "good", "bad", "great", "high",
               "low", "quick", "slow", "fast", "strong", "weak",
               "beautiful", "pretty", "ugly", "rich", "poor", "happy",
               "sad", "angry", "quiet", "loud", "clean", "dirty",
               "hot", "cold", "warm", "cool", "dark", "bright", "soft",
               "hard", "heavy", "light", "easy", "difficult", "simple",
               "complex", "empty", "full", "open", "closed", "real",
               "fake", "whole", "half", "various", "several", "few",
               "many", "much", "most", "more", "less", "other", "same",
               "different", "famous", "important", "interesting",
               "wonderful", "terrible", "delicious", "hungry", "thirsty",
               "tired", "sleepy", "awake", "ready", "late", "early",
               "right", "wrong", "true", "false", "certain", "sure",
               "clear", "vague", "deep", "shallow", "wide", "narrow",
               "thick", "thin", "fat", "skinny", "sweet", "sour",
               "bitter", "salty", "sharp", "dull", "smooth", "rough",
               "sticky", "slippery", "wet", "dry", "fresh", "stale",
               "raw", "cooked", "frozen", "boiling", "expensive",
               "cheap", "modern", "ancient", "wooden", "plastic",
               "metal", "golden", "silver", "royal", "main", "chief",
               "principal", "major", "minor", "central", "local",
               "foreign", "domestic", "public", "private", "personal",
               "general", "special", "particular", "specific", "usual",
               "normal", "strange", "weird", "odd", "funny", "serious",
               "gentle", "rough", "polite", "rude", "kind", "cruel",
               "brave", "cowardly", "proud", "modest", "wise", "foolish",
               "clever", "stupid", "smart", "dumb", "crazy", "sane",
               "healthy", "sick", "ill", "dead", "alive", "blind",
               "deaf", "mute", "lame", "naked", "bare", "bald",
               "hairy", "furry", "feathery", "scaly", "slimy", "dusty",
               "muddy", "rusty", "shiny", "glossy", "faded", "worn",
               "torn", "broken", "fixed", "newborn", "teenage",
               "middle-aged", "elderly", "adult", "infant", "junior",
               "senior", "northern", "southern", "eastern", "western",
               "upper", "lower", "inner", "outer", "front", "back",
               "side", "top", "bottom", "left", "right", "middle",
               "near", "far", "distant", "close", "adjacent", "neighboring"}


def _next_content(toks: List[str], i: int) -> Optional[str]:
    """Nächstes Nicht-Adjektiv/Determiner-Wort ab Position i+1."""
    for j in range(i + 1, len(toks)):
        if toks[j] not in _ADJECTIVES:
            return toks[j]
    return None


def rule_determiner_noun(toks: List[str]) -> int:
    """R3: this/that + Plural-Nomen und these/those + Singular-Nomen
    (über zwischenstehende Adjektive hinweg)."""
    for i, t in enumerate(toks):
        if t in ("this", "that"):
            nxt = _next_content(toks, i)
            if nxt is None:
                continue
            return -1 if _is_plural(nxt) else 1
        if t in ("these", "those"):
            nxt = _next_content(toks, i)
            if nxt is None:
                continue
            return -1 if not _is_plural(nxt) else 1
    return 0


def rule_subject_verb(toks: List[str]) -> int:
    """R4: Subjekt-Numerus vs. Verb-Form (3sg).
    In wh-Fragen steht das Subjekt NACH dem Auxiliar: 'What senators
    was Alicia approaching?' — was kongruiert mit Alicia."""
    # wh-Fragen-Zweig
    if toks and toks[0] in _WH_WORDS:
        for i, t in enumerate(toks[1:], start=1):
            if t in ("is", "was", "has", "does"):
                subj = toks[i + 1] if i + 1 < len(toks) else ""
                if not subj:
                    return 0
                return -1 if _is_plural(subj) else 1
            if t in ("are", "were", "have", "do"):
                subj = toks[i + 1] if i + 1 < len(toks) else ""
                if not subj:
                    return 0
                return 1 if _is_plural(subj) else -1
        return 0
    # Subjekt = erstes Inhaltswort nach einem Determiner (oder Satzanfang),
    # das kein Verb-Signal trägt; Verb = das direkt folgende Wort.
    # Relativsatz-Subjekte (nach who/that/which) sind Distraktoren —
    # überspringen; bei folgendem Relativsatz ist das Hauptverb das
    # letzte Verb-ähnliche Token.
    subj_idx, verb_idx = None, None
    skip = {"the", "a", "an", "this", "that", "these", "those", "my",
            "your", "his", "her", "its", "our", "their"}
    for i, t in enumerate(toks):
        if t in skip:
            continue
        if t in ("is", "are", "was", "were", "has", "have", "had"):
            # Hilfsverb zuerst -> Subjekt davor suchen
            subj_idx = i - 1
            verb_idx = i
            break
        if _is_plural(t) or t in _SINGULAR_IRREG or t in _PLURAL_IRREG:
            # Distraktor-Schutz: Subjekt in einem Relativsatz überspringen
            if i >= 2 and toks[i - 1] in ("who", "that", "which", "whom"):
                continue
            if i >= 3 and toks[i - 2] in ("who", "that", "which", "whom"):
                continue
            subj_idx = i
            if i + 1 < len(toks):
                if toks[i + 1] in ("who", "that", "which", "whom"):
                    # Relativsatz folgt: Hauptverb = letztes Verb-Token
                    for j in range(len(toks) - 1, i, -1):
                        if _is_3sg(toks[j]) or toks[j] in (
                                "run", "eat", "play", "go", "walk",
                                "sleep", "work", "live", "seem", "suffer",
                                "cooperate", "read", "write", "speak",
                                "sing", "drink", "help", "love", "know",
                                "see", "want", "need", "worry", "worry"):
                            verb_idx = j
                            break
                else:
                    verb_idx = i + 1
            break
    if subj_idx is None or verb_idx is None:
        return 0
    subj, verb = toks[subj_idx], toks[verb_idx]
    if verb not in (_3SG_IRREG | {"are", "have", "were"} | _PLURAL_PRON):
        # Verb-Kandidat muss verb-typisch sein: 3sg oder Pluralform
        if not (_is_3sg(verb) or verb in ("run", "eat", "play", "go", "walk",
                                          "sleep", "work", "live", "seem",
                                          "become", "remain", "stay", "sit",
                                          "stand", "read", "write", "speak",
                                          "sing", "drink", "help", "love",
                                          "know", "see", "want", "need")):
            return 0
    subj_plural = _is_plural(subj)
    verb_3sg = _is_3sg(verb) or verb in _3SG_IRREG
    if subj_plural and verb_3sg:
        return -1
    if not subj_plural and not verb_3sg and verb not in ("are", "were", "have"):
        return -1
    return 1


# Unregelmäßige Partizipien (für R10)
_PARTICIPLES = {"worn", "hidden", "gone", "come", "seen", "taken",
                "given", "broken", "written", "eaten", "spoken",
                "driven", "ridden", "frozen", "chosen", "stolen",
                "woken", "torn", "borne", "sworn", "beaten", "bitten",
                "forgotten", "fallen", "blown", "flown", "drawn",
                "thrown", "grown", "known", "shown", "shaken", "done",
                "begun", "drunk", "sung", "swum", "run", "risen",
                "written", "brought", "bought", "caught", "taught",
                "thought", "fought", "sought", "found", "held", "kept",
                "left", "lost", "made", "met", "paid", "said", "sold",
                "sent", "spent", "stood", "told", "understood", "won"}
# Unregelmäßige einfache Vergangenheitsformen (für die Attributiv-Regel)
_PAST_IRREG = {"wore", "hid", "went", "came", "saw", "took", "gave",
               "broke", "wrote", "ate", "spoke", "drove", "rode",
               "froze", "chose", "stole", "woke", "tore", "bore",
               "swore", "beat", "bit", "forgot", "fell", "blew",
               "flew", "drew", "threw", "grew", "knew", "shook",
               "showed", "ran", "rose", "began", "drank", "sang",
               "swam", "brought", "bought", "caught", "taught",
               "thought", "fought", "sought", "found", "held", "kept",
               "left", "lost", "made", "met", "paid", "said", "sold",
               "sent", "spent", "stood", "told", "understood", "won"}
_AUX_BEFORE = {"have", "has", "had", "having", "is", "are", "was",
               "were", "be", "been", "being", "get", "gets", "got",
               "gotten", "become", "becomes"}
_DET = {"the", "a", "an", "some", "any", "all", "both", "each",
        "every", "no", "my", "your", "his", "her", "its", "our",
        "their", "this", "that", "these", "those"}


def _attributive(toks: List[str], i: int) -> bool:
    """Partizip/Vergangenheit in attributiver Position? Attributiv heißt:
    Determiner UNMITTELBAR davor ('the hidden offspring'). Das trennt
    'the Borgias worn' (Prädikat, det bei i-2) von 'the hidden glass'
    (Attribut, det bei i-1) und 'Sandra known Becca' (Prädikat)."""
    return i > 0 and toks[i - 1] in _DET


def rule_participle(toks: List[str]) -> int:
    """R10: Partizip als Hauptverb ohne Auxiliar = Verletzung
    ('The Borgias worn scarves' vs 'wore'). Attributiv (vor Nomen) ist
    das Partizip ein korrektes Adjektiv ('the hidden offspring');
    einfache Vergangenheit attributiv ist die Verletzung ('the hid
    offspring')."""
    for i, t in enumerate(toks):
        if t in _PARTICIPLES:
            if _attributive(toks, i):
                return 1  # Partizip-Adjektiv: korrekt
            if i > 0 and toks[i - 1] in _AUX_BEFORE:
                return 1  # Perfekt/Passiv mit Auxiliar: korrekt
            return -1  # Prädikats-Partizip ohne Auxiliar: Verletzung
        if t in _PAST_IRREG:
            if _attributive(toks, i):
                return -1  # Vergangenheit als Adjektiv: Verletzung
    return 0


_WH_WORDS = {"whose", "what", "which", "who", "whom", "where", "when",
              "why", "how"}
_AUX = {"had", "has", "have", "is", "are", "was", "were", "should",
        "would", "could", "can", "may", "might", "must", "does", "did",
        "do", "will"}

# Verben, die ein Objekt verlangen (für die that-Gap-Regel)
_GAP_VERBS = {"conceal", "conceals", "concealed", "concealing", "examine",
              "examined", "examining", "reveal", "revealed", "revealing",
              "scare", "scared", "scaring", "drive", "drove", "driving",
              "hide", "hid", "hidden", "hiding", "show", "showed",
              "shown", "showing", "tell", "told", "telling", "visit",
              "visited", "visiting", "heal", "healed", "healing",
              "alarm", "alarmed", "alarming", "wear", "wore", "worn",
              "wearing", "notice", "noticed", "noticing", "teach",
              "taught", "teaching", "feed", "fed", "feeding", "follow",
              "followed", "following", "help", "helped", "helping",
              "see", "saw", "seen", "seeing", "hear", "heard",
              "hearing", "find", "found", "finding", "like", "liked",
              "loving", "love", "loved", "hate", "hated", "fear",
              "feared", "fearing", "eat", "ate", "eaten", "eating",
              "read", "reading", "write", "wrote", "written", "writing"}


def rule_wh_island(toks: List[str]) -> int:
    """R5: Satzstart wh-Wort + direkt Auxiliar = Insel-Verletzung
    (left branch island: 'What is Renee concealing movies?' vs
    'What movies is Renee concealing?')."""
    if len(toks) < 2 or toks[0] not in _WH_WORDS:
        return 0
    return -1 if toks[1] in _AUX else 1


# Präpositionen, die eine Verb-Lücke abschließen können ("flee from")
_PREPS = {"from", "with", "to", "at", "about", "on", "in", "for", "by",
          "of", "into", "through", "toward", "upon"}
# Artikel/Skopus-Wörter am Satzende, die keine Objekte sind
_TAIL_SKIP = _PREPS | {"the", "a", "an", "some", "any", "all", "both",
                       "each", "every"}
# Matrix-Verben, die einen Komplementsatz einleiten ("realize that ...")
_MATRIX_VERBS = {"realize", "realizes", "realized", "remember",
                 "remembers", "remembered", "know", "knows", "knew",
                 "discover", "discovers", "discovered", "forget",
                 "forgets", "forgot", "figure", "figured", "learn",
                 "learns", "learned", "see", "sees", "saw", "seen",
                 "think", "thinks", "thought", "say", "says", "said",
                 "believe", "believes", "believed", "find", "finds",
                 "found", "hear", "hears", "heard", "notice", "notices",
                 "noticed", "conceal", "conceals", "concealed",
                 "recall", "recalls", "recalled", "understand",
                 "understands", "understood"}


def _complementizer_pos(toks: List[str], w: str) -> int:
    """Position des letzten 'that'/'who' — nur wenn es nach einem
    Matrix-Verb steht (Komplementizer), sonst -1 (Relativpronomen)."""
    for i in range(len(toks) - 1, -1, -1):
        if toks[i] == w and i > 0 and toks[i - 1] in _MATRIX_VERBS:
            return i
    return -1


def rule_that_gap(toks: List[str]) -> int:
    """R6: Komplementsatz mit Lücke (transitives Verb ohne Objekt, ggf. +
    Präposition) nach 'that' = Verletzung; nach 'who/what' = ok.
    Relativsätze ('the book that John read') sind KEINE Verletzung —
    das 'that' folgt dort einem Nomen, nicht einem Matrix-Verb."""
    if len(toks) < 4:
        return 0
    # Lücke: letzter Inhalts-Token ist ein Gap-Verb
    tail = [t for t in toks[-4:] if t not in _TAIL_SKIP]
    if not tail or tail[-1] not in _GAP_VERBS:
        return 0
    that_pos = _complementizer_pos(toks, "that")
    if that_pos >= 0:
        return -1
    for w in ("who", "what", "whom", "whose", "which"):
        if _complementizer_pos(toks, w) >= 0:
            return 1
    return 0


# Auxiliare für die NPI-Lizenzierungs-Regel
_AUX_NPI = {"have", "has", "had", "did", "does", "do", "can", "could",
            "might", "may", "will", "would", "should", "must", "is",
            "are", "was", "were"}
_NPI_WORDS = {"ever", "any"}
_NEG = {"not", "n't", "never", "no"}


def rule_npi_licensing(toks: List[str]) -> int:
    """R9: NPI ('ever'/'any') braucht eine Negation zwischen dem letzten
    Auxiliar und dem NPI ('have not ever' ok; 'have ever' Verletzung —
    die Negation in einem Nebensatz lizenziert nicht)."""
    for i, t in enumerate(toks):
        if t not in _NPI_WORDS:
            continue
        # letztes Auxiliar vor dem NPI suchen
        aux = None
        for j in range(i - 1, -1, -1):
            if toks[j] in _AUX_NPI:
                aux = j
                break
            if toks[j] in ("that", "which", "who", "whom", ",", "."):
                break
        if aux is None:
            continue
        neg = any(k in _NEG for k in toks[aux + 1:i])
        if not neg:
            return -1
        return 1
    return 0


def rule_aux_agreement(toks: List[str]) -> int:
    """R11: Auxiliar kongruiert mit dem KOPF-Nomen, nicht mit dem
    Relativsatz-Distraktor: 'This customer who had visited most children
    HAS worn' (customer = singular) vs '...children HAVE worn'.
    Subjekt = Token VOR dem letzten Relativpronomen vor dem Auxiliar."""
    _REL = {"who", "that", "which", "whom"}
    # Auxiliare inkl. kontrahierter Formen (aren't -> 'aren')
    for i, t in enumerate(toks):
        if t not in ("has", "have", "does", "do", "is", "are", "was",
                     "were", "aren", "isnt", "wasnt", "werent",
                     "doesnt", "didnt", "hasnt", "havent"):
            continue
        # letztes Relativpronomen vor dem Auxiliar suchen
        rel = -1
        for j in range(i - 1, -1, -1):
            if toks[j] in _REL:
                rel = j
                break
            if toks[j] == ",":
                break
        if rel <= 0:
            continue
        subj = toks[rel - 1]
        if subj in ("that", "which", "who", "the"):
            continue
        plural = _is_plural(subj) or subj in _PLURAL_IRREG
        sg_aux = t in ("has", "does", "is", "was", "isnt", "wasnt",
                       "doesnt", "hasnt")
        pl_aux = t in ("have", "do", "are", "were", "aren", "werent",
                       "havent")
        if sg_aux and plural:
            return -1
        if pl_aux and not plural:
            return -1
        return 1
    return 0


# tough-movement vs. raising-Adjektive (geschlossene Lexikon-Klassen)
_TOUGH_ADJ = {"pleasant", "easy", "interesting", "ready", "hard",
              "difficult", "fun", "nice", "important", "impossible",
              "possible", "good", "bad", "dangerous", "safe",
              "convenient", "enjoyable", "tough", "simple", "exciting",
              "boring", "tedious", "useful", "helpful", "worthwhile"}
_RAISING_ADJ = {"apt", "certain", "likely", "bound", "sure", "soon",
                "unlikely", "liable", "due", "expected", "supposed",
                "destined", "fated", "doomed"}


def rule_tough_raising(toks: List[str]) -> int:
    """R12: 'X is ADJ to V' — die Struktur entscheidet:
      'to V' OHNE Objekt (hängend)  -> tough-Adjektiv korrekt
        ('James is pleasant to flee from')
      'to V' MIT Objekt             -> raising-Adjektiv korrekt
        ('Rachel was apt to talk to Alicia')
    Variante 2 von tough_vs_raising ist genau invertiert zu Variante 1 —
    die Objekt-Struktur ist der Diskriminator."""
    for i, t in enumerate(toks):
        if t not in _TOUGH_ADJ and t not in _RAISING_ADJ:
            continue
        if i + 1 >= len(toks) or toks[i + 1] != "to":
            continue
        is_tough = t in _TOUGH_ADJ
        # Objekt nach 'to V'? toks[i+2] = Verb, toks[i+3] = Objekt-Kandidat
        if i + 3 < len(toks):
            obj = toks[i + 3]
            has_obj = obj not in (_PREPS | _DET | _AUX_BEFORE | _ADJECTIVES |
                                  {"to", "that", "which", "who", "and",
                                   "or", "but", "not", "the"})
            if has_obj:
                return 1 if not is_tough else -1
        # kein Objekt -> tough-Struktur
        return 1 if is_tough else -1
    return 0


def _has_clause_boundary(toks: List[str]) -> bool:
    return any(t in ("that", "which", "whom", "who") for t in toks)


def _anaphor_subject(toks: List[str], reflex_pos: int) -> Optional[str]:
    """Antezedens eines Reflexivs:
    - MIT Satzgrenze (that/who/which): Hauptsatz-Subjekt (c_command —
      Objekte wie 'couch' in 'sell some couch' sind keine Subjekte)
    - OHNE Grenze (verschachtelt, 'imagine campuses are boring'):
      das nächste [NP]+Verb-Subjekt (Domäne 2/3)"""
    if _has_clause_boundary(toks):
        return _main_subject(toks[:reflex_pos] + toks[reflex_pos + 1:]) or \
            _main_subject(toks)
    return _nearest_subject(toks, reflex_pos) or _main_subject(toks)


def _nearest_subject(toks: List[str], reflex_pos: int) -> Optional[str]:
    """Das NÄCHSTE Subjekt vor dem Reflexiv (Principle A, Domäne 2/3):
    [NP] direkt vor einem Verb, NICHT nach einer Präposition ('about
    Gregory') und nicht über that/who-Grenzen. 'campuses are boring
    themselves' -> campuses, nicht Donald."""
    i = reflex_pos - 1
    while i >= 0:
        t = toks[i]
        if t in ("that", "which", "whom", "who"):
            break
        if i > 0 and toks[i - 1] in _PREPS:
            i -= 1
            continue
        if t in _SKIP_SUBJ or t in _AUX_WORDS or t in _REFLEXIVES:
            i -= 1
            continue
        # [NP] + Verb/aux danach?
        if i + 1 < len(toks) and (
                toks[i + 1] in _AUX_WORDS or toks[i + 1] in _GAP_VERBS or
                toks[i + 1] in _PAST_IRREG or
                toks[i + 1].endswith("ed")):
            return t
        i -= 1
    return None


_REFLEXIVES = {"herself", "himself", "themselves", "itself", "myself",
                "ourselves", "yourselves"}
_PRONOUNS = {"her", "him", "them", "it", "me", "us", "you"}


def rule_reflexive_domain(toks: List[str]) -> int:
    """R13 (Principle A, Domäne): Ein Reflexiv im NEBENSATZ kann nicht an
    das Hauptsatz-Subjekt binden ('Carla explained that Samuel discussed
    HERSELF' = Verletzung; '...discussed her' = ok). Ein Reflexiv nach dem
    Hauptsatz-Verb (zweites Verb nach der Grenze) ist normal und wird von
    den anderen Anapher-Regeln behandelt."""
    boundary = -1
    for i, t in enumerate(toks):
        if t in ("that", "which", "whom", "who"):
            boundary = i
    if boundary < 0:
        return 0
    if _main_subject(toks[:boundary]) is None:
        return 0
    verbs_after = 0
    for t in toks[boundary + 1:]:
        if t in _AUX_WORDS or t in ("has", "have", "had"):
            continue
        if t in _GAP_VERBS or t in _PARTICIPLES or t in _PAST_IRREG or \
                t.endswith("ed") or t.endswith("s"):
            verbs_after += 1
        if t in _REFLEXIVES:
            if verbs_after <= 1:
                return -1  # Reflexiv im Nebensatz: Domän-Verletzung
            return 0       # Hauptsatz: andere Regeln entscheiden
        if t in _PRONOUNS and verbs_after <= 1:
            return 1       # Pronomen im Nebensatz: normale Bindung
    return 0


RULES = [
    ("anaphor_gender", rule_anaphor_gender),
    ("anaphor_number", rule_anaphor_number),
    ("determiner_noun", rule_determiner_noun),
    ("subject_verb", rule_subject_verb),
    ("aux_agreement", rule_aux_agreement),
    ("tough_raising", rule_tough_raising),
    ("wh_island", rule_wh_island),
    ("that_gap", rule_that_gap),
    ("npi_licensing", rule_npi_licensing),
    ("participle", rule_participle),
    ("reflexive_domain", rule_reflexive_domain),
]


# ---------------------------------------------------------------------------
# WinoGrande: Pronomen-Kongruenz über den Blank
# ---------------------------------------------------------------------------

_PRON_MALE = {"he", "him", "his", "himself"}
_PRON_FEMALE = {"she", "her", "hers", "herself"}
_PRON_PLURAL = {"they", "them", "their", "themselves"}
_PRON_SING = {"it", "its", "itself"}


def _option_gender(opt: str) -> Optional[str]:
    """Genus einer Option: male/female über Namen+Rollen, sonst None."""
    toks = _toks(opt)
    for t in toks:
        if t in _FEMALE_SIGNALS:
            return "female"
        if t in _MALE_SIGNALS:
            return "male"
    return None


def winogrande_rule(sentence: str, option1: str, option2: str) -> Optional[int]:
    """Kongruenz-Entscheidung: 1/2 wenn eine Option eindeutig passt,
    None wenn keine Regel anwendbar (dann LM-Backoff)."""
    # Tokenisierung MIT Blank (Unterstrich als Token erhalten)
    toks = re.findall(r"[a-z_]+", str(sentence).lower())
    blank = None
    for i, t in enumerate(toks):
        if t == "_":
            blank = i
            break
    if blank is None:
        return None
    o1g, o2g = _option_gender(option1), _option_gender(option2)
    o1p, o2p = _is_plural(option1), _is_plural(option2)

    # nächstes Genus/Numerus-Pronomen um den Blank (Fenster 8)
    for i in range(max(0, blank - 8), min(len(toks), blank + 9)):
        if i == blank:
            continue
        t = toks[i]
        if t in _PRON_MALE:
            if o1g == "male" and o2g != "male":
                return 1
            if o2g == "male" and o1g != "male":
                return 2
        if t in _PRON_FEMALE:
            if o1g == "female" and o2g != "female":
                return 1
            if o2g == "female" and o1g != "female":
                return 2
        if t in _PRON_PLURAL:
            if o1p and not o2p:
                return 1
            if o2p and not o1p:
                return 2
        if t in _PRON_SING and o1p != o2p:
            # 'it' + Singular-Verb-Signal -> Singular-Option
            return 1 if not o1p else 2
    return None


# Namen sind im Englischen großgeschrieben — die Case-Info des
# Originalsatzes ist der wahre Name-Diskriminator (kein Raten über
# unbekannte Wörter: 'couch' ist kein Name, 'Karla' ist einer).
_raw_caps: set = set()


def apply_rules(sentence: str) -> List[Tuple[str, int]]:
    """Alle Regeln auf einen Satz anwenden: [(name, +1/-1/0), ...]."""
    global _raw_caps
    _raw_caps = set(t.lower() for t in re.findall(r"[A-Z][a-z]+", sentence))
    toks = _toks(sentence)
    return [(name, fn(toks)) for name, fn in RULES]


def structural_score(sentence: str) -> int:
    """Summe der anwendbaren Regeln (positiv = strukturell sauber)."""
    return sum(v for _, v in apply_rules(sentence))
