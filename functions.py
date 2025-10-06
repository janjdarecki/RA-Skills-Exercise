import re, unicodedata
import pandas as pd
import numpy as np


SURNAME_PARTICLES = {"de","da","di","del","della","der","van","von","la","le","du","dos","das","do","of","the"}


SUFFIX_TOKENS = {"jr","sr","iii","ii","iv"}


NICKNAMES = {
    "al": "alfred", "amo": "amory", "bernie": "bernard", "bernard": "ben", "bernard": "bennie",
    "bill": "william", "billy": "william", "will": "william",
    "bob": "robert", "bobby": "robert", "rob": "robert", "robin": "robert",
    "rick": "richard", "ricky": "richard", "rickey": "richard", "dick": "richard",
    "jim": "james", "jimmy": "james", "jamie": "james", "jimmie": "james", "jimbo": "james",
    "joe": "joseph", "joey": "joseph",
    "frank": "francis", "frankie": "francis",
    "tony": "anthony", "ann": "annie",
    "tom": "thomas", "tommy": "thomas",
    "liz": "elizabeth", "beth": "elizabeth", "betty": "elizabeth",
    "kate": "katherine", "katie": "katherine", "kathy": "katherine",
    "steve": "steven", "stevie": "steven", "stephen": "steve",
    "deborah": "debbie", "cynthia": "cindy",
    "jon": "jonathan", "johnny": "john", "johnnie": "john", "jack": "john",
    "mike": "michael", "mikey": "michael",
    "chris": "christopher", "pete": "peter",
    "phil": "philip", "phillip": "philip",
    "greg": "gregory", "gregg": "gregory",
    "andy": "andrew", "drew": "andrew",
    "ben": "benjamin", "sam": "samuel",
    "ed": "edward", "eddie": "edward", "edmund": "edward",
    "ted": "theodore", "teddy": "theodore",
    "mel": "melvin", "edolphus": "ed",
    "larry": "lawrence", "laurie": "lawrence",
    "jeff": "jeffrey", "jeffy": "jeffrey",
    "ray": "raymond", "ron": "ronald", "ronnie": "ronald",
    "charlie": "charles", "chuck": "charles",
    "maggie": "margaret", "peggy": "margaret", "meg": "margaret", "marge": "margaret",
    "hank": "henry", "gus": "augustus",
    "wally": "wallace", "bud": "william",
    "sue": "susan", "susie": "susan", "suzie": "susan",
    "jenny": "jennifer", "jen": "jennifer",
    "jessie": "jessica", "jess": "jessica",
    "pat": "patrick", "paddy": "patrick",
    "terry": "terence", "terri": "teresa", "terryanne": "teresa",
    "toni": "antonia", "tonia": "antonia",
    "josh": "joshua", "nate": "nathaniel", "nat": "nathaniel",
    "dan": "daniel", "danny": "daniel",
    "don": "donald", "donnie": "donald",
    "doug": "douglas", "alexander": "alex",
    "helen": "helena",
    "marty": "martin", "art": "arthur", "artie": "arthur",
    "lenny": "leonard", "len": "leonard", "leo": "leonard",
    "hal": "harold", "harry": "harold",
    "claud": "claude",
    "cliff": "clifford", "cliffy": "clifford",
    "tim": "timothy", "timmy": "timothy",
    "del": "wendell", "zachary": "zach", "zachary": "zack",
    "wesley": "wes", "sander": "sandy", "richard": "rich",
    "richard": "ric", "randall": "randy", "patricia": "pat",
    "patricia":"patsy", "patsy": "pat", "nicholas": "nick",
    "melquiades": "mel", "melton": "mel", "michael": "mac",
    "mike": "mac", "herbert": "herb", "gilbert": "gil",
    "gerald": "gerry", "ernest": "ernie", "elijah": "eli",
    "david": "dave", "curtis": "curt", "bradley": "brad"
}


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    

def normalize_text(s: str) -> str:
    s = _strip_accents(str(s or ""))
    s = re.sub(r"\bo['’](\w+)", r"o\1", s, flags=re.IGNORECASE)
    s = "".join(" " if unicodedata.category(ch) == "Pd" else ch for ch in s)
    tbl = dict.fromkeys(map(ord, "’'.,;:()[]{}/\\|&+*`~!@#$%^_=<>?\""), " ")
    s = s.translate(tbl)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def tokens(s: str) -> list:
    return normalize_text(s).split()


def is_initial(tok: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z]\.?", tok or ""))


def first_token_keep_noninitials(name: str) -> str:
    s = str(name or "").strip()
    if not s: return ""
    parts = [p.strip(" ,") for p in re.split(r"\s+", s) if p.strip(" ,")]
    i = 0
    while i < len(parts) and is_initial(parts[i]):
        i += 1
    if i >= len(parts): return ""  # only initials
    tok = normalize_text(parts[i])
    return tok.split()[0] if tok else ""


def ordered_words(s: str) -> list:
    return tokens(s)


def surname_tokens_from_last(last_name: str) -> list:
    s = normalize_text(last_name)
    s = re.sub(r"\bo['’](\w+)", r"o\1", s)
    raw = s.split()
    keep = [t for t in raw if t not in SURNAME_PARTICLES]
    return keep or raw


def surname_tokens_from_fullname(name: str) -> list:
    s = normalize_text(name)
    s = re.sub(r"\bo['’](\w+)", r"o\1", s)
    words = s.split()
    if len(words) <= 1: return []
    return [w for w in words[1:] if w not in SURNAME_PARTICLES]


def extract_suffix(words: list) -> set:
    return {w for w in words[-3:] if w in SUFFIX_TOKENS}


def expand_firstname_options(token: str) -> list:
    if not token: return []
    t = token.strip().lower()
    rev = {}
    for nick, canon in NICKNAMES.items():
        rev.setdefault(canon, []).append(nick)
    out = {t}
    if t in NICKNAMES: out.add(NICKNAMES[t])
    if t in rev: out.update(rev[t])
    return list(out)
    

def firstname_variant_pool(first_name: str, full_name: str, last_name: str) -> list:
    fn_toks  = tokens(first_name)
    name_toks= tokens(full_name)
    last_set = set(surname_tokens_from_last(last_name))
    base_given = [
        t for t in (fn_toks + name_toks)
        if t and not is_initial(t)
        and t not in last_set
        and t not in SUFFIX_TOKENS        # ← exclude suffixes
    ]
    base_given = list(dict.fromkeys(base_given))
    pool = set()
    for t in base_given:
        pool.update(expand_firstname_options(t))
    return list(pool)


def middle_initials_from_first(first_name: str) -> set:
    raw = str(first_name or "").strip()
    if not raw: return set()
    parts = re.split(r"\s+", raw)
    i = 0
    while i < len(parts) and is_initial(parts[i]):
        i += 1
    mids = parts[i+1:] if i < len(parts) else []
    out = set()
    for m in mids:
        m = m.strip(" ,")
        if not m: continue
        if is_initial(m):
            out.add(m[0].lower())
        else:
            w = normalize_text(m)
            if w:
                out.add(w[0])
    return out