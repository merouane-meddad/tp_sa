"""Regular expressions and small lexicons shared by preprocessing and metadata.

Keeping them in one place guarantees that Task 1, the model tokenizer and
Task 3 detect URLs, emoticons, etc. in exactly the same way.
"""
import re

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"(?<!\w)@\w+")
HASHTAG_RE = re.compile(r"(?<!\w)#(\w+)")

# ASCII emoticons. They must be searched on text WITHOUT URLs: the previous
# pattern also matched ":/" in every "https://" and "8p" / "8D" inside t.co
# hashes, which turned "emoticon" into a copy of "has URL".
_EYES = r"[:;=]"
_NOSE = r"[-o^']?"
EMO_POS_RE = re.compile(
    rf"(?<![:;=<\\]){_EYES}{_NOSE}[)\]D}}pP3*](?![\w/:;])"   # :) :-) :D ;) :P :3
    rf"|(?<!\S)[(\[]{_NOSE}{_EYES}(?!\S)"                    # (:  [:
    r"|(?<!\S)[xX]D+(?!\S)"                                  # xD XD
    r"|(?<![<\w])<3+(?![\w/])"                               # <3
    r"|\^[_.]?\^"                                            # ^_^ ^^
)
EMO_NEG_RE = re.compile(
    rf"(?<![:;=<\\]){_EYES}'?{_NOSE}[(\[/\\|{{](?![\w/:;])"  # :( :'( :/ :| :\
    rf"|(?<!\S)[)\]]{_NOSE}{_EYES}(?!\S)"                     # ):  ]:
    rf"|(?<!\S)D{_NOSE}{_EYES}(?!\S)"                         # D:
    r"|</3+"                                                  # </3
    r"|-_-|T_T|;_;"
)

# A few frequent emoji with an unambiguous conventional polarity.
EMOJI_POS = set("😀😁😃😄😆😊😍😘😗😙😚☺🙂🤗😎🥰😻💕💖💗💓💞💘❤♥💯👍👏🙌🎉🎊✨😉😋😜😛🤩")
EMOJI_NEG = set("😞😟😠😡😢😣😥😦😧😨😩😫😭😰😱😒😔😕🙁☹😤💔👎🙄🤬🤮😖😓")

# Censored insults. A candidate is a token mixing letters and masking symbols
# (f**k, sh*t, b!tch, a$$); it only counts if, once the symbols are treated as
# wildcards, it matches a known swear word. This rejects "A$AP", "is@the",
# "great**" or "to#nfl", which a symbol-only regex would accept.
CENSOR_CANDIDATE_RE = re.compile(r"(?<![\w@#])[a-z]*[*#@$%!]+[a-z*#@$%!]*(?![\w@#])", re.IGNORECASE)
MASKED_WORD_RE = re.compile(r"(?<!\w)[a-z]-word\b", re.IGNORECASE)    # "f-word", "n-word"
SWEAR_WORDS = (
    "fuck", "fucks", "fucked", "fucker", "fucking", "fuckin", "shit", "shitty",
    "bitch", "bitches", "ass", "asses", "asshole", "damn", "nigga", "niggas",
    "nigger", "piss", "pissed", "pissing", "dick", "crap", "bastard", "hell",
    "cunt", "whore", "slut", "motherfucker",
)


def is_censored_swear(token: str) -> bool:
    """True if ``token`` is a masked version of a swear word (f**k -> fuck)."""
    token = token.lower().strip("!")
    if token[:1] in ("#", "@"):        # hashtags and mentions, not masking
        return False
    if not re.search(r"[*#@$%!]", token) or not re.search(r"[a-z]", token):
        return False
    pattern = re.compile("".join("." if c in "*#@$%!" else re.escape(c) for c in token))
    return any(len(w) == len(token) and pattern.fullmatch(w) for w in SWEAR_WORDS)


# Uncensored swear words, for comparison with their censored forms.
SWEAR_RE = re.compile(
    r"\b(fuck\w*|shit\w*|bitch\w*|ass(?:hole)?s?|damn\w*|crap\w*|bastard\w*|"
    r"dick\w*|piss\w*|wtf|stfu|idiot\w*|moron\w*|stupid)\b",
    re.IGNORECASE,
)

# Explicit irony / sarcasm markers. Words such as "great" or "sure" are NOT
# used on their own: they are far more often literal than ironic.
IRONY_RE = re.compile(
    r"#(?:sarcasm|sarcastic|irony|ironic|not)\b|(?<!\w)/s\b|"
    r"\b(?:yeah,? right|thanks a lot|just what i needed|oh,? great|"
    r"what a surprise|how convenient|so much fun|love it when|"
    r"i just love how|i love how|said no one ever)\b",
    re.IGNORECASE,
)
# "Scare quotes": a short quoted expression of 1 to 3 words.
SCARE_QUOTES_RE = re.compile(r'"(?:[^"\s]+(?:\s[^"\s]+){0,2})"')

ELONGATED_RE = re.compile(r"\b\w*(\w)\1{2,}\w*\b")       # soooo, nooooo
CAPS_WORD_RE = re.compile(r"\b[A-Z][A-Z']{2,}\b")        # GREAT, NEVER
