import numpy as np
import os
import re
import json
import imageio.v3 as iio
import cv2
from scipy import ndimage
from skimage import exposure
import time
import urllib.request
import urllib.parse
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# SINGLETON OCR ENGINE MANAGEMENT
# ============================================================

class OCREngines:
    """
    Lazy-initialized singleton holder for OCR engines.
    Created once, passed into all processing functions.
    """
    def __init__(self):
        self._easyocr_reader = None
        self._yomitoku_ocr = None
        self._yomitoku_available = None

    @property
    def easyocr(self):
        if self._easyocr_reader is None:
            import easyocr
            print("  [init] Loading EasyOCR (ja+en)...")
            t = time.time()
            self._easyocr_reader = easyocr.Reader(['ja', 'en'], gpu=False, verbose=False)
            print(f"  [init] EasyOCR ready in {time.time()-t:.1f}s")
        return self._easyocr_reader

    @property
    def yomitoku(self):
        if self._yomitoku_ocr is None:
            if self._yomitoku_available is False:
                return None
            try:
                from yomitoku import OCR as YomitokuOCR
                print("  [init] Loading Yomitoku...")
                t = time.time()
                self._yomitoku_ocr = YomitokuOCR(device="cpu")
                print(f"  [init] Yomitoku ready in {time.time()-t:.1f}s")
                self._yomitoku_available = True
            except ImportError:
                print("  [init] Yomitoku not installed — JP fallback to EasyOCR")
                self._yomitoku_available = False
                return None
            except Exception as e:
                print(f"  [init] Yomitoku init failed: {e}")
                self._yomitoku_available = False
                return None
        return self._yomitoku_ocr

    @property
    def has_yomitoku(self):
        if self._yomitoku_available is None:
            # Trigger init to find out
            _ = self.yomitoku
        return self._yomitoku_available is True


# ============================================================
# CONSTANTS
# ============================================================

CARD_RATIO = 88.9 / 63.5

PREFIXES = ['メガ', 'Ｍ', 'かがやく', 'ヒスイの', 'ヒスイ', 'ガラルの', 'ガラル',
            'パルデアの', 'パルデア', 'アローラの', 'アローラ', 'げんし']
SUFFIXES = ['EX', 'ex', 'ＥＸ', 'GX', 'ＧＸ', 'V', 'VMAX', 'VSTAR', 'δ',
            'CR', 'ＣＲ', 'M', 'Ｍ', 'X', 'Ｘ', 'CX', 'BX', 'FX', 'eX', 'gX']

RARITY_CODES = {
    'C': 'Common', 'U': 'Uncommon', 'R': 'Rare', 'RR': 'Double Rare',
    'RRR': 'Triple Rare', 'SR': 'Special Rare', 'SAR': 'Special Art Rare',
    'UR': 'Ultra Rare', 'AR': 'Art Rare', 'MA': 'Master', 'H': 'Holo',
    'HR': 'Hyper Rare', 'CHR': 'Character Rare', 'CSR': 'Character Super Rare',
    'S': 'Special', 'A': 'Amazing', 'K': 'Kira', 'TR': 'Trainer Rare',
    'PR': 'Promo',
}

BOTTOM_KEYWORDS = ['pokemon', 'nintendo', 'creatures', 'game freak', 'freak',
                   '.co.jp', '©', 'illus', 'ilus', 'kodama', 'promo']
BOTTOM_JP_KEYWORDS = ['抵抗力', 'にげる', 'にける']

POKEMON_KEYWORDS = ['pokemon', 'nintendo', 'creatures', 'game freak', '©',
                    'ポケモン', 'エネルギー', 'ダメージ', '特性', 'HP', 'hp',
                    'ex', 'EX', 'GX', 'VMAX', 'にげる', '抵抗力', 'ワザ']

SUPABASE_URL = os.environ.get('SUPABASE_URL', '')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY', '')

# ============================================================
# OCR IMAGE DOWNSCALING
# ============================================================

MAX_OCR_HEIGHT = 900  # Downscale cards taller than this before OCR


def downscale_for_ocr(image):
    """Downscale image if too tall, return (scaled_image, scale_factor)."""
    h, w = image.shape[:2]
    if h <= MAX_OCR_HEIGHT:
        return image, 1.0
    scale = MAX_OCR_HEIGHT / h
    new_w = int(w * scale)
    new_h = MAX_OCR_HEIGHT
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def run_easyocr(reader, image, oh, ow):
    """Run EasyOCR and return normalized detections list."""
    img_small, scale = downscale_for_ocr(image)
    sh, sw = img_small.shape[:2]
    results = reader.readtext(img_small, detail=1)
    detections = []
    for (bbox, text, conf) in results:
        # Normalize against the small image dims — gives 0-1 coords
        cy = np.mean([p[1] for p in bbox]) / sh
        cx = np.mean([p[0] for p in bbox]) / sw
        h_frac = abs(bbox[2][1] - bbox[0][1]) / sh
        detections.append({'cy': cy, 'cx': cx, 'text': text,
                           'conf': conf, 'h_frac': h_frac})
    return detections


def run_yomitoku(yomi_engine, image, oh, ow):
    """Run Yomitoku and return normalized detections list."""
    if yomi_engine is None:
        return []
    try:
        img_small, scale = downscale_for_ocr(image)
        sh, sw = img_small.shape[:2]
        # Convert to BGR (OpenCV format)
        ocr_bgr = cv2.cvtColor(img_small, cv2.COLOR_RGB2BGR) if img_small.ndim == 3 else img_small
        result, _ = yomi_engine(ocr_bgr)
        detections = []
        if hasattr(result, 'words') and len(result.words) > 0:
            for w in result.words:
                text = ''
                for attr in ['content', 'text', 'value']:
                    if hasattr(w, attr):
                        text = getattr(w, attr)
                        break
                cy, cx = 0.5, 0.5
                for pts_attr in ['points', 'box', 'bounding_box']:
                    if hasattr(w, pts_attr):
                        pts = np.array(getattr(w, pts_attr))
                        if pts.ndim == 2 and pts.shape[1] >= 2:
                            cy = pts[:, 1].mean() / sh
                            cx = pts[:, 0].mean() / sw
                            break
                score = 0.0
                for score_attr in ['rec_score', 'score', 'confidence', 'conf']:
                    if hasattr(w, score_attr):
                        score = float(getattr(w, score_attr))
                        break
                if text:
                    detections.append({'cy': cy, 'cx': cx, 'text': str(text),
                                       'score': score})
        return detections
    except Exception as e:
        print(f"  [!] Yomitoku error: {e}")
        return []


def detect_language_fast(engines, ocr_img, oh, ow):
    """Quick language detection from name region crop.
    Runs EasyOCR on a tiny strip (~12% of image) — much faster than full image."""
    name_crop = ocr_img[int(oh * 0.03):int(oh * 0.13), 0:int(ow * 0.65)]
    crop_h, crop_w = name_crop.shape[:2]
    if crop_h < 10 or crop_w < 10:
        return 'en'
    results = engines.easyocr.readtext(name_crop, detail=1)
    all_text = ''.join(r[1] for r in results)
    jp_chars = sum(1 for c in all_text if ord(c) > 0x3000)
    return 'ja' if jp_chars >= 2 else 'en'



# ============================================================
# FIELD EXTRACTION
# ============================================================

def find_text_in_region(detections, y_min, y_max, x_min, x_max):
    return ' '.join(d['text'] for d in detections
                    if y_min <= d['cy'] <= y_max and x_min <= d['cx'] <= x_max).strip()


def find_all_in_region(detections, y_min, y_max, x_min, x_max):
    return [d for d in detections
            if y_min <= d['cy'] <= y_max and x_min <= d['cx'] <= x_max]


def extract_hp(detections, y_min, y_max, x_min, x_max):
    text = find_text_in_region(detections, y_min, y_max, x_min, x_max)
    hp_m = re.search(r'(\d{2,4})', text)
    return text, hp_m.group(1) if hp_m else ''


def extract_card_number(detections, y_min, y_max, x_min, x_max):
    text = find_text_in_region(detections, y_min, y_max, x_min, x_max)
    m = re.search(r'(\d{1,4})\s*/\s*(\d{1,4})', text)
    if m:
        return text, m.group(1).strip(), m.group(0).replace(' ', '')
    m = re.search(r'(\d{1,4})\s*/\s*([A-Za-z][A-Za-z0-9-]+)', text)
    if m:
        return text, m.group(1).strip(), m.group(0).replace(' ', '')
    return text, '', ''


def extract_name(detections, y_min, y_max, x_min, x_max):
    candidates = find_all_in_region(detections, y_min, y_max, x_min, x_max)
    if not candidates:
        return ''

    # Evolution markers that appear near the name but ARE NOT the name
    EVOLUTION_MARKERS = ['から進化', 'からたねポケモン', 'からの進化', '進化']

    # Junk patterns to skip entirely
    JUNK_PATTERNS = [r'^[-=─━]+$', r'^[\s\-\.]+$']

    best = ''
    best_score = -1
    for d in candidates:
        t = d['text'].strip()

        # Skip empty or very short
        if len(t) < 2:
            continue

        # Skip junk patterns
        if any(re.match(p, t) for p in JUNK_PATTERNS):
            continue

        # Clean trailing junk (dashes, dots, spaces)
        t_clean = re.sub(r'[\s\-\=─━\.]+$', '', t).strip()
        if len(t_clean) < 2:
            continue

        # Check if this is an evolution marker like "ハクリューから進化"
        is_evolution_text = any(marker in t_clean for marker in EVOLUTION_MARKERS)

        katakana_count = sum(1 for c in t_clean if '\u30A0' <= c <= '\u30FF')
        latin_count = sum(1 for c in t_clean if c.isascii() and c.isalpha())
        score = katakana_count * 2 + latin_count

        # Penalize very long text (probably description, not name)
        if len(t_clean) > 20:
            score -= len(t_clean)

        # Heavy penalty for evolution text — it should almost never win
        if is_evolution_text:
            score -= 20

        if score > best_score:
            best_score = score
            best = t_clean

    # Final cleanup: strip evolution suffixes if they somehow got through
    for marker in ['から進化', 'からの進化']:
        if best.endswith(marker):
            best = best[:-len(marker)].strip()

        # Also handle "Xから進化" where X is embedded
        idx = best.find(marker)
        if idx > 0:
            best = best[:idx].strip()

    return best


def extract_fields(detections, label=""):
    """Extract name, HP, card number from a set of detections."""
    MIN_CONF = 0.15
    filtered = [d for d in detections
                if float(d.get('conf', d.get('score', 0))) >= MIN_CONF]
    fields = {}

    # Name extraction uses ALL detections (unfiltered) because card names on
    # holographic/foil cards often have very low OCR confidence due to reflective text.
    # The extract_name function handles quality via its own scoring logic.
    fields['name'] = extract_name(detections, 0.04, 0.12, 0.0, 0.65)

    # HP and card number use filtered detections (higher confidence needed for numbers)
    hp_text, hp_num = extract_hp(filtered, 0.0, 0.12, 0.5, 1.0)
    fields['hp'] = hp_text
    fields['hp_number'] = hp_num

    set_text, card_num, full_id = extract_card_number(filtered, 0.85, 1.0, 0.0, 1.0)
    fields['set_code'] = set_text
    fields['card_number'] = full_id

    # Fallback: wider name region (also unfiltered)
    if not fields['name']:
        fields['name'] = extract_name(detections, 0.0, 0.20, 0.0, 0.70)
        if fields['name']:
            print(f"  [{label} fallback] Name from wider top (0-20%)")

    # Fallback: wider HP region
    if not fields['hp_number']:
        hp_text, hp_num = extract_hp(filtered, 0.0, 0.20, 0.4, 1.0)
        if hp_num:
            fields['hp'] = hp_text
            fields['hp_number'] = hp_num
            print(f"  [{label} fallback] HP from wider region (0-20%): {hp_num}")

    # Extra-wide HP for sleeved cards
    if not fields['hp_number']:
        hp_text, hp_num = extract_hp(filtered, 0.0, 0.30, 0.3, 1.0)
        if hp_num:
            fields['hp'] = hp_text
            fields['hp_number'] = hp_num
            print(f"  [{label} fallback] HP from extra-wide region (0-30%): {hp_num}")

    # Fallback: wider card number region
    if not fields['card_number']:
        set_text, card_num, full_id = extract_card_number(filtered, 0.80, 1.0, 0.0, 1.0)
        if card_num:
            fields['set_code'] = set_text
            fields['card_number'] = full_id

    # Last resort: full scan for card number
    if not fields['card_number']:
        for det in detections:
            m = re.search(r'(\d{1,4})\s*/\s*([A-Za-z0-9-]+)', det['text'])
            if m:
                fields['card_number'] = m.group(0).replace(' ', '')
                fields['set_code'] = det['text']
                print(f"  [{label} full-scan] Card#: {fields['card_number']}")
                break

    # Last resort: full scan for name (katakana)
    if not fields['name']:
        for det in sorted(detections, key=lambda d: d['cy']):
            if det['cy'] > 0.45:
                break
            t = det['text']
            katakana_count = sum(1 for c in t if '\u30A0' <= c <= '\u30FF')
            if katakana_count >= 3 and len(t) <= 15:
                fields['name'] = t
                print(f"  [{label} full-scan] Name candidate: '{t}'")
                break

    fields['all_text'] = detections
    return fields


# ============================================================
# POKEMON NAME MATCHING
# ============================================================

def load_pokemon_db(image_path):
    """Load pokemon name database from pokemonnames.ts."""
    pokemon_names_path = os.path.join(os.path.dirname(image_path), 'pokemonnames.ts')
    pokemon_db = {}
    if not os.path.exists(pokemon_names_path):
        return pokemon_db
    with open(pokemon_names_path, 'r', encoding='utf-8') as f:
        ts_content = f.read()

    json_str = ts_content.strip()
    for wrapper in ['export default ', 'export const pokemonNames = ',
                    'module.exports = ', 'const pokemonNames = ']:
        if json_str.startswith(wrapper):
            json_str = json_str[len(wrapper):]
            break
    json_str = json_str.rstrip('; \n\t')
    if json_str.endswith('as const'):
        json_str = json_str[:-len('as const')].rstrip()
    arr_start = json_str.find('[')
    arr_end = json_str.rfind(']')
    if arr_start >= 0 and arr_end > arr_start:
        json_str = json_str[arr_start:arr_end + 1]
    try:
        for entry in json.loads(json_str):
            jp_name = entry.get('jp', '').strip()
            if jp_name:
                pokemon_db[jp_name] = {
                    'en': entry.get('en', ''),
                    'dex': int(entry.get('pokedex', 0) or 0),
                    'hiragana': entry.get('jphiragana', ''),
                    'suffix': entry.get('suffix', '') or '',
                    'prefix': entry.get('prefix', '') or '',
                }
    except json.JSONDecodeError:
        pass
    return pokemon_db


def load_pokemon_db_from_json(json_path):
    """Load pokemon name database from bundled JSON file."""
    pokemon_db = {}
    if not os.path.exists(json_path):
        print(f"  [!] Pokemon DB not found: {json_path}")
        return pokemon_db
    with open(json_path, 'r', encoding='utf-8') as f:
        entries = json.load(f)
    for entry in entries:
        jp_name = entry.get('jp', '').strip()
        if jp_name:
            pokemon_db[jp_name] = {
                'en': entry.get('en', ''),
                'dex': int(entry.get('pokedex', 0) or 0),
                'hiragana': entry.get('jphiragana', ''),
                'suffix': entry.get('suffix', '') or '',
                'prefix': entry.get('prefix', '') or '',
            }
    return pokemon_db


def strip_pokemon_affixes(name):
    core = name.strip()
    for p in sorted(PREFIXES, key=len, reverse=True):
        if core.startswith(p):
            core = core[len(p):]
            break
    for s in sorted(SUFFIXES, key=len, reverse=True):
        if core.endswith(s):
            core = core[:-len(s)]
            break
    return core.strip()


def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[len(s2)]


def _fuzzy_match_single(core, db):
    """Try exact, substring, then Levenshtein match for a single core name."""
    if not core:
        return None
    if core in db:
        info = db[core]
        return {'matched_name': core, 'core': core, 'dist': 0, **info, 'method': 'exact'}

    for db_name, info in db.items():
        if len(core) >= 2 and len(db_name) >= 2:
            if core in db_name or db_name in core:
                return {'matched_name': db_name, 'core': core, 'dist': 0,
                        **info, 'method': 'substring'}

    best, best_d, best_info = None, 999, None
    max_dist = 3
    for db_name, info in db.items():
        if abs(len(db_name) - len(core)) > max_dist:
            continue
        d = levenshtein(core, db_name)
        if d <= max_dist and d < best_d:
            best, best_d, best_info = db_name, d, info

    if best:
        return {'matched_name': best, 'core': core, 'dist': best_d,
                **best_info, 'method': f'fuzzy(d={best_d})'}
    return None


def extract_pokemon_names(ocr_name, db):
    """Find all known Pokemon names embedded in the OCR text.
    Returns list of (db_name, info) tuples found in the text."""
    found = []
    text = ocr_name
    # Check longest DB names first to avoid partial matches
    for db_name in sorted(db.keys(), key=len, reverse=True):
        if len(db_name) >= 2 and db_name in text:
            found.append((db_name, db[db_name]))
            # Remove matched name to avoid double-matching substrings
            text = text.replace(db_name, '', 1)
    return found


def fuzzy_match_pokemon(ocr_name, db):
    core = strip_pokemon_affixes(ocr_name)
    if not core:
        return None

    # Multi-pokemon cards (& in name): extract all Pokemon names first
    # Must run BEFORE single-name match to capture all names
    if '&' in core:
        found_names = extract_pokemon_names(core, db)
        if found_names:
            db_name, info = found_names[0]
            method = 'embedded'
            if len(found_names) > 1:
                method = f'embedded (&: {"+".join(n for n, _ in found_names)})'
            return {'matched_name': db_name, 'core': core, 'dist': 0, **info,
                    'method': method,
                    'all_pokemon_names': [n for n, _ in found_names]}

    # Try full name (single pokemon)
    result = _fuzzy_match_single(core, db)
    if result:
        return result

    # Last resort: scan for embedded names even without &
    found_names = extract_pokemon_names(core, db)
    if found_names:
        db_name, info = found_names[0]
        return {'matched_name': db_name, 'core': core, 'dist': 0, **info,
                'method': 'embedded',
                'all_pokemon_names': [n for n, _ in found_names]}

    return None


# ============================================================
# CARD ID PARSING
# ============================================================

def parse_card_id(set_code_text, card_number='', set_size=''):
    """Parse rarity from set_code text. Accepts pre-extracted card_number and set_size
    from extract_fields to avoid redundant regex parsing."""
    result = {'card_number': card_number, 'set_size': set_size,
              'rarity_code': '', 'rarity_name': ''}
    if not set_code_text:
        return result

    # If card_number/set_size weren't pre-supplied, extract them
    if not card_number:
        m = re.search(r'(\d{1,4})\s*/\s*(\d{1,4})', set_code_text)
        if m:
            result['card_number'] = m.group(1)
            result['set_size'] = m.group(2)
        else:
            m = re.search(r'(\d{1,4})\s*/\s*([A-Za-z][A-Za-z0-9-]+)', set_code_text)
            if m:
                result['card_number'] = m.group(1)
                result['set_size'] = m.group(2)
            else:
                return result

    # Find rarity code after the card number pattern
    m = re.search(r'\d{1,4}\s*/\s*[\dA-Za-z][A-Za-z0-9-]*', set_code_text)
    after = set_code_text[m.end():].strip() if m else set_code_text.strip()

    for code in sorted(RARITY_CODES.keys(), key=len, reverse=True):
        if after.upper().startswith(code):
            result['rarity_code'] = code
            result['rarity_name'] = RARITY_CODES[code]
            break

    if not result['rarity_code']:
        rm = re.match(r'([A-Za-z]{1,3})', after)
        if rm:
            result['rarity_code'] = rm.group(1).upper()
            result['rarity_name'] = RARITY_CODES.get(
                result['rarity_code'], f'Unknown ({result["rarity_code"]})')

    promo_m = re.search(r'\b(PROMO|PR)\b', set_code_text, re.IGNORECASE)
    if promo_m and not result['rarity_code']:
        result['rarity_code'] = 'PR'
        result['rarity_name'] = 'Promo'

    return result


# ============================================================
# SUPABASE + SIFT
# ============================================================

def supabase_query(filters):
    params = ['select=*']
    for col, val in filters.items():
        if isinstance(val, tuple):  # Support custom operators: ('ilike', '*keyword*')
            op, operand = val
            params.append(f"{col}={op}.{urllib.parse.quote(str(operand), safe='*')}")
        else:
            params.append(f"{col}=eq.{urllib.parse.quote(str(val), safe='')}")
    url = f"{SUPABASE_URL}/rest/v1/cards?{'&'.join(params)}"
    req = urllib.request.Request(url, headers={
        'apikey': SUPABASE_KEY,
        'Authorization': f'Bearer {SUPABASE_KEY}',
        'Content-Type': 'application/json',
    })
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        print(f"  [!] Query error: {e}")
        return []


def download_image(url, timeout=10):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            img_bytes = resp.read()
            img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception:
        pass
    return None


def sift_match_score(photo_kp, photo_desc, ref_gray, sift_eng, clahe_eng, photo_shape):
    """SIFT match using pre-computed photo keypoints/descriptors."""
    ph, pw = photo_shape[:2]
    ref_resized = cv2.resize(ref_gray, (pw, ph), interpolation=cv2.INTER_AREA)
    ref_eq = clahe_eng.apply(ref_resized)

    kp2, d2 = sift_eng.detectAndCompute(ref_eq, None)
    if photo_desc is None or d2 is None or len(photo_kp) < 2 or len(kp2) < 2:
        return 0, 0, 0.0
    raw = cv2.BFMatcher(cv2.NORM_L2).knnMatch(photo_desc, d2, k=2)
    n_good = sum(1 for m, n in raw if m.distance / (n.distance + 1e-10) < 0.75)
    return n_good, len(raw), n_good / max(len(raw), 1)


def region_sift_score(photo_region_kp_desc, ref_gray_full, oh, ow,
                      sift_eng, clahe_eng, region_specs):
    """SIFT region matching using pre-computed photo region keypoints."""
    total_good, total_total = 0, 0
    for key, (ry0f, ry1f, rx0f, rx1f) in region_specs.items():
        kp_r, desc_r, rh, rw = photo_region_kp_desc[key]
        if rh == 0 or rw == 0:
            continue

        ref_crop = ref_gray_full[int(oh * ry0f):int(oh * ry1f),
                                 int(ow * rx0f):int(ow * rx1f)]
        ref_crop = cv2.resize(ref_crop, (rw, rh), interpolation=cv2.INTER_AREA)
        rg = clahe_eng.apply(ref_crop)

        kp2, d2 = sift_eng.detectAndCompute(rg, None)
        if desc_r is not None and d2 is not None and len(kp_r) >= 2 and len(kp2) >= 2:
            raw = cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc_r, d2, k=2)
            good = sum(1 for m, n in raw if m.distance / (n.distance + 1e-10) < 0.75)
            total_good += good
            total_total += len(raw)

    return total_good, total_total, total_good / max(total_total, 1)


# ============================================================
# STAGE 1: BLOB DETECTION (unchanged logic)
# ============================================================

def _make_blob_entry(mask, blob_id):
    """Build a blob dict from a binary mask. Returns None if invalid."""
    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(contour)
    rect_center, rect_size, rect_angle = rect
    rect_w, rect_h = rect_size
    if min(rect_w, rect_h) < 50:
        return None
    ratio = max(rect_w, rect_h) / min(rect_w, rect_h)
    is_card_like = 1.1 < ratio < 1.7
    return {
        'blob_id': blob_id,
        'mask': mask,
        'contour': contour,
        'rect': rect,
        'rect_center': rect_center,
        'rect_size': rect_size,
        'rect_angle': rect_angle,
        'rect_w': rect_w,
        'rect_h': rect_h,
        'ratio': ratio,
        'is_card_like': is_card_like,
        'area': float(mask.sum()),
    }


def _run_morphology(enhanced, sigma, usize, close_kernel, close_iters):
    """Run the blur → threshold → morphology → label pipeline with given params."""
    blurred = ndimage.gaussian_filter(enhanced, sigma=sigma)
    local_mean = ndimage.uniform_filter(blurred, size=usize)
    local_diff = np.abs(blurred - local_mean)
    edge_mask = local_diff > 0.04
    closed = ndimage.binary_closing(edge_mask,
                                    structure=np.ones((close_kernel, close_kernel)),
                                    iterations=close_iters)
    cleaned = ndimage.binary_opening(closed, structure=np.ones((3, 3)), iterations=1)
    filled = ndimage.binary_fill_holes(cleaned)
    labeled, n_features = ndimage.label(filled)
    return filled, labeled, n_features


def _collect_blobs(labeled, n_features, filled, min_area):
    """Collect valid blobs from a labeled image."""
    if n_features == 0:
        return []
    sizes = ndimage.sum(filled, labeled, range(1, n_features + 1))
    blobs = []
    for blob_id in range(1, n_features + 1):
        if sizes[blob_id - 1] < min_area:
            continue
        mask = (labeled == blob_id)
        b = _make_blob_entry(mask, blob_id)
        if b is not None:
            blobs.append(b)
    blobs.sort(key=lambda b: b['area'], reverse=True)
    return blobs


def detect_all_blobs(image_path):
    """
    Find all card-like blobs using a two-pass strategy:
    Pass 1 (light): sigma=3, usize=31 — preserves inter-card gaps in grids
    Pass 2 (heavy): sigma=7, usize=51 — robust for single cards on busy backgrounds
    Pick whichever pass yields more card-like blobs.
    """
    raw_img = iio.imread(image_path, mode='L').astype(float) / 255.0
    color_img = iio.imread(image_path)
    enhanced = exposure.rescale_intensity(raw_img)
    total_pixels = raw_img.shape[0] * raw_img.shape[1]
    min_area = total_pixels * 0.02

    # --- Pass 1: light parameters (good for multi-card grids) ---
    filled_l, labeled_l, n_l = _run_morphology(enhanced, sigma=3, usize=31,
                                                close_kernel=3, close_iters=2)
    blobs_light = _collect_blobs(labeled_l, n_l, filled_l, min_area)
    n_card_light = sum(1 for b in blobs_light if b['is_card_like'])

    # --- Pass 2: heavy parameters (original — good for single/few cards) ---
    filled_h, labeled_h, n_h = _run_morphology(enhanced, sigma=7, usize=51,
                                                close_kernel=5, close_iters=3)
    blobs_heavy = _collect_blobs(labeled_h, n_h, filled_h, min_area)
    n_card_heavy = sum(1 for b in blobs_heavy if b['is_card_like'])

    print(f"Light pass: {len(blobs_light)} blobs, {n_card_light} card-like")
    print(f"Heavy pass: {len(blobs_heavy)} blobs, {n_card_heavy} card-like")

    # Pick the pass that finds more card-like blobs
    if n_card_light >= n_card_heavy:
        blobs = blobs_light
        print(f"→ Using light pass ({n_card_light} card-like blobs)")
    else:
        blobs = blobs_heavy
        print(f"→ Using heavy pass ({n_card_heavy} card-like blobs)")

    print(f"Found {len(blobs)} blobs:")
    for i, b in enumerate(blobs):
        print(f"  [{i+1}] area={b['area']:.0f} ratio={b['ratio']:.3f} card_like={b['is_card_like']}")

    return blobs, raw_img, color_img, enhanced


def detect_all_blobs_from_array(color_img):
    """detect_all_blobs but from an in-memory numpy array (no file I/O)."""
    # Use luminance-weighted grayscale (same as iio.imread mode='L')
    raw_img = (0.299 * color_img[:,:,0].astype(float) +
               0.587 * color_img[:,:,1].astype(float) +
               0.114 * color_img[:,:,2].astype(float)) / 255.0
    enhanced = exposure.rescale_intensity(raw_img)
    total_pixels = raw_img.shape[0] * raw_img.shape[1]
    min_area = total_pixels * 0.02

    filled_l, labeled_l, n_l = _run_morphology(enhanced, sigma=3, usize=31,
                                                close_kernel=3, close_iters=2)
    blobs_light = _collect_blobs(labeled_l, n_l, filled_l, min_area)
    n_card_light = sum(1 for b in blobs_light if b['is_card_like'])

    filled_h, labeled_h, n_h = _run_morphology(enhanced, sigma=7, usize=51,
                                                close_kernel=5, close_iters=3)
    blobs_heavy = _collect_blobs(labeled_h, n_h, filled_h, min_area)
    n_card_heavy = sum(1 for b in blobs_heavy if b['is_card_like'])

    if n_card_light >= n_card_heavy:
        blobs = blobs_light
    else:
        blobs = blobs_heavy

    return blobs, raw_img, color_img, enhanced


# ============================================================
# STAGE 2A: PERSPECTIVE WARP (separated from orientation)
# ============================================================

def dewarp_card(blob, color_img, enhanced):
    """Perspective warp a blob into a portrait card image.
    Returns (ocr_img_color, ocr_gray, oh, ow) or None if invalid."""
    rect = blob['rect']
    rect_center, rect_size, rect_angle = rect
    rect_w, rect_h = rect_size

    # --- Perspective warp ---
    box_pts = cv2.boxPoints(rect)
    box_pts_sorted = box_pts[np.argsort(box_pts[:, 1])]
    top_two = box_pts_sorted[:2]
    bot_two = box_pts_sorted[2:]
    tl, tr = top_two[np.argsort(top_two[:, 0])]
    bl, br = bot_two[np.argsort(bot_two[:, 0])]
    src_corners = np.array([tl, tr, br, bl], dtype=np.float32)

    dst_short = min(rect_w, rect_h)
    dst_long = dst_short * CARD_RATIO
    side_top = np.linalg.norm(tr - tl)
    side_left = np.linalg.norm(bl - tl)
    if side_left > side_top:
        dst_w, dst_h = int(dst_short), int(dst_long)
    else:
        dst_w, dst_h = int(dst_short), int(dst_long)
        src_corners = np.array([tr, br, bl, tl], dtype=np.float32)

    dst_corners = np.array([
        [0, 0], [dst_w, 0], [dst_w, dst_h], [0, dst_h]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_corners, dst_corners)
    img_gray_u8 = (enhanced * 255).astype(np.uint8)
    photo_dewarped = cv2.warpPerspective(img_gray_u8, M, (dst_w, dst_h))
    photo_dewarped_color = cv2.warpPerspective(color_img, M, (dst_w, dst_h))

    # Force portrait
    ocr_img = photo_dewarped_color.copy()
    ocr_gray = photo_dewarped.copy()
    oh, ow = ocr_img.shape[:2]
    if ow > oh:
        ocr_img = cv2.rotate(ocr_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        ocr_gray = cv2.rotate(ocr_gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
        oh, ow = ocr_img.shape[:2]

    return ocr_img, ocr_gray, oh, ow


# ============================================================
# STAGE 2B: ORIENTATION DETECTION (fast corner-crop approach)
# ============================================================

# Anchor patterns to look for in quadrants
# EasyOCR reads upside-down "HP" as "dH", "dh", "Hd", "qH" etc
_HP_PATTERN = re.compile(r'HP|hp|Hp|hP|dH|dh|Hd|hd|qH|pH')
_HP_NUM_PATTERN = re.compile(r'\d{2,3}')
_TRAINER_PATTERNS = ['トレーナーズ', 'トレーナー', 'trainers', 'trainer',
                     'Trainers', 'Trainer', 'TRAINERS', 'グッズ', 'サポート',
                     'スタジアム', 'supporter', 'stadium', 'item']
# Upside-down HP pattern (dH + numbers = HP read upside down)
# EasyOCR commonly reads upside-down "HP" as: dH, dh, Hd, qH, pH, aH, oH
_UPSIDE_HP_PATTERN = re.compile(r'dH|dh|Hd|hd|qH|pH|aH|oH|dh|DH')


def _check_quadrant_for_anchor(reader, image, oh, ow, y0f, y1f, x0f, x1f, label=""):
    """Run EasyOCR on a quadrant crop and check for HP or Trainer text.
    Returns: (type_str_or_None, confidence, raw_text, results_list)"""
    y0, y1 = int(oh * y0f), int(oh * y1f)
    x0, x1 = int(ow * x0f), int(ow * x1f)
    crop = image[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    if ch < 10 or cw < 10:
        return None, 0, '', [], crop

    results = reader.readtext(crop, detail=1)
    all_text = ' '.join(r[1] for r in results)
    avg_conf = float(np.mean([r[2] for r in results])) if results else 0

    # Log what was read
    if results:
        texts = [f"'{r[1]}'({r[2]:.2f})" for r in results[:5]]
        print(f"    [{label}] {ch}×{cw}px → {', '.join(texts)}")
    else:
        print(f"    [{label}] {ch}×{cw}px → (no text detected)")

    # Check for upright HP (normal reading)
    has_upright_hp = bool(re.search(r'HP|hp|Hp|hP|aH', all_text))
    # Check for upside-down HP — EasyOCR reads "HP" upside down as "dH", "dh", etc
    has_upside_hp = bool(_UPSIDE_HP_PATTERN.search(all_text))
    has_numbers = bool(_HP_NUM_PATTERN.search(all_text))

    if has_upright_hp and has_numbers:
        return 'hp', avg_conf, all_text, results, crop
    if has_upright_hp:
        return 'hp_only', avg_conf, all_text, results, crop
    if has_upside_hp and has_numbers:
        return 'hp_upside', avg_conf, all_text, results, crop
    if has_upside_hp:
        return 'hp_upside', avg_conf, all_text, results, crop

    # Check for trainer keywords
    for kw in _TRAINER_PATTERNS:
        if kw in all_text:
            return 'trainer', avg_conf, all_text, results, crop

    return None, avg_conf, all_text, results, crop


def orient_card(blob, color_img, enhanced, engines, blob_index):
    """
    Fast orientation detection using 2 corner crops.

    Strategy: Check top-right (HP area if upright) and bottom-left (HP area if flipped).
    - Upright HP in top-right → keep
    - Upside-down HP (dH) in bottom-left → flip
    - No anchors found → ambiguous (use majority vote)
    """
    print(f"\n  [orient {blob_index+1}] ratio={blob['ratio']:.3f} angle={blob['rect_angle']:.1f}°")

    # --- Perspective warp ---
    ocr_img, ocr_gray, oh, ow = dewarp_card(blob, color_img, enhanced)

    t0 = time.time()
    reader = engines.easyocr

    # ── Only 2 quadrant regions (saves ~50% EasyOCR time) ──
    TR = (0.00, 0.20, 0.45, 1.00)  # top-right strip: HP area
    BL = (0.80, 1.00, 0.00, 0.55)  # bottom-left strip: mirror of TR

    tr_type, tr_conf, tr_text, tr_res, tr_crop = _check_quadrant_for_anchor(
        reader, ocr_img, oh, ow, *TR, label="TopRight")
    bl_type, bl_conf, bl_text, bl_res, bl_crop = _check_quadrant_for_anchor(
        reader, ocr_img, oh, ow, *BL, label="BotLeft")

    # Score: positive = correct orientation, negative = flipped
    score = 0

    # TopRight: upright HP → correct
    if tr_type in ('hp', 'hp_only'):
        score += 10
        print(f"    ✓ Top-right: {tr_type} → UPRIGHT")
    elif tr_type == 'hp_upside':
        score += 5
        print(f"    ~ Top-right: upside-down HP text (unusual)")
    elif tr_type == 'trainer':
        score += 10
        print(f"    ✓ Top-right: trainer → UPRIGHT")

    # BotLeft: any HP signal → card is flipped
    if bl_type in ('hp', 'hp_only', 'hp_upside'):
        score -= 10
        print(f"    ✓ Bottom-left: {bl_type} → FLIPPED")
    elif bl_type == 'trainer':
        score -= 10
        print(f"    ✓ Bottom-left: trainer → FLIPPED")

    elapsed = time.time() - t0

    if score > 0:
        needs_flip = False
    elif score < 0:
        needs_flip = True
    else:
        needs_flip = None
        print(f"    [!] No anchor found — ambiguous")

    print(f"  [orient {blob_index+1}] score={score:+d} → "
          f"{'FLIP' if needs_flip == True else ('KEEP' if needs_flip == False else 'AMBIG')} "
          f"({elapsed:.1f}s)")

    return {
        'ocr_img_0': ocr_img,
        'ocr_gray_0': ocr_gray,
        'oh': oh,
        'ow': ow,
        'orient_score': score,
    }


def apply_orientation(orient_data, flipped):
    """Pick the correct orientation's image. Computes 180° rotation lazily if needed."""
    if flipped:
        ocr_img = cv2.rotate(orient_data['ocr_img_0'], cv2.ROTATE_180)
        ocr_gray = cv2.rotate(orient_data['ocr_gray_0'], cv2.ROTATE_180)
        return ocr_img, ocr_gray
    else:
        return (orient_data['ocr_img_0'],
                orient_data['ocr_gray_0'])


# ============================================================
# STAGE 2C: PROCESS SINGLE BLOB (with pre-determined orientation)
# ============================================================

def process_single_blob(blob, color_img, enhanced, clahe, image_path,
                        blob_index, engines, pokemon_db,
                        orient_data, card_flipped, yomitoku_lock=None):
    """Process a single detected blob using pre-determined orientation.
    Runs full OCR ONCE on the correctly oriented image.
    yomitoku_lock: if provided, serialize Yomitoku access for thread safety."""
    ratio = blob['ratio']
    is_card = blob['is_card_like']

    print(f"\n{'='*60}")
    print(f"CARD {blob_index + 1}: ratio={ratio:.3f} flipped={card_flipped}")
    print(f"{'='*60}")

    ocr_img, ocr_gray = apply_orientation(orient_data, card_flipped)
    oh = orient_data['oh']
    ow = orient_data['ow']

    # --- Detect language (cheap mini-crop), then run full OCR ONCE ---
    # Mini-crop costs ~0.5-1.5s but avoids a full EasyOCR pass (~2-4s) on JP cards.
    def _run_ocr():
        lang = detect_language_fast(engines, ocr_img, oh, ow)
        t0 = time.time()
        if lang == 'ja' and engines.has_yomitoku:
            detections = run_yomitoku(engines.yomitoku, ocr_img, oh, ow)
            if detections:
                return detections, lang, 'yomitoku', time.time() - t0
            # Yomitoku returned nothing — fall back to EasyOCR
        detections = run_easyocr(engines.easyocr, ocr_img, oh, ow)
        return detections, lang, 'easyocr', time.time() - t0

    # Serialize OCR engine access for thread safety (single code path)
    if yomitoku_lock:
        with yomitoku_lock:
            detections, lang, ocr_engine_used, elapsed = _run_ocr()
    else:
        detections, lang, ocr_engine_used, elapsed = _run_ocr()

    print(f"  {ocr_engine_used}: {len(detections)} detections in {elapsed:.1f}s")

    for det in sorted(detections, key=lambda d: d['cy']):
        conf_val = float(det.get('conf', det.get('score', 0)))
        marker = '✓' if conf_val >= 0.15 else '✗'
        print(f"  {marker} [{det['cy']:.2f},{det['cx']:.2f}] "
              f"'{det['text'][:40]}' (conf={conf_val:.2f})")

    print(f"  Language: {lang} | Engine: {ocr_engine_used}")

    # --- Extract fields from the REUSED detections (no new OCR) ---
    primary_fields = extract_fields(detections, label=ocr_engine_used.capitalize())

    # For JP cards with EasyOCR fallback (no Yomitoku), note it
    if lang == 'ja' and ocr_engine_used == 'easyocr':
        print(f"  (Yomitoku unavailable, using EasyOCR for JP)")

    ocr_fields = dict(primary_fields)

    # --- Card ID parsing (pass pre-extracted fields to avoid redundant regex) ---
    # Extract card_number digits from the full_id (e.g. "045/078" → "045")
    card_num_from_fields = ''
    set_size_from_fields = ''
    full_id = ocr_fields.get('card_number', '')
    if full_id:
        m = re.match(r'(\d{1,4})\s*/\s*(\S+)', full_id)
        if m:
            card_num_from_fields = m.group(1)
            set_size_from_fields = m.group(2)

    card_id = parse_card_id(
        ocr_fields.get('set_code', ''),
        card_number=card_num_from_fields,
        set_size=set_size_from_fields)
    rarity_str = (f" rarity={card_id['rarity_code']}({card_id['rarity_name']})"
                  if card_id.get('rarity_code') else "")
    print(f"  Fields: name='{ocr_fields.get('name','')}' "
          f"HP={ocr_fields.get('hp_number','')} "
          f"card#={ocr_fields.get('card_number','')}{rarity_str}")

    is_japanese = any(ord(c) > 0x3000 for c in ocr_fields.get('name', ''))

    # --- Pokemon name matching ---
    ocr_name_candidates = []
    if ocr_fields.get('name'):
        ocr_name_candidates.append(('primary', ocr_fields['name']))

    pokemon_match = None
    for source, raw_name in ocr_name_candidates:
        match = fuzzy_match_pokemon(raw_name, pokemon_db)
        if match:
            match['source'] = source
            match['raw_ocr'] = raw_name
            pokemon_match = match
            break

    # --- SIFT setup: single engine, CLAHE photo ONCE ---
    sift = cv2.SIFT_create(nfeatures=2000)

    region_specs = {
        'name':     (0.04, 0.12, 0.03, 0.60),
        'hp':       (0.00, 0.12, 0.50, 0.97),
        'set_code': (0.85, 1.00, 0.03, 0.97),
    }

    # CLAHE the full photo ONCE — slice regions from the equalized result
    photo_gray_for_sift = (cv2.cvtColor(ocr_img, cv2.COLOR_RGB2GRAY)
                           if ocr_img.ndim == 3 else ocr_img)
    photo_gray_eq = clahe.apply(photo_gray_for_sift)
    photo_kp, photo_desc = sift.detectAndCompute(photo_gray_eq, None)

    # Pre-compute region keypoints from already-equalized full image
    photo_region_kp_desc = {}
    regions_for_viz = {}
    for key, (ry0f, ry1f, rx0f, rx1f) in region_specs.items():
        region_crop = ocr_img[int(oh * ry0f):int(oh * ry1f),
                              int(ow * rx0f):int(ow * rx1f)]
        regions_for_viz[key] = region_crop
        rh, rw = region_crop.shape[:2]
        if rh == 0 or rw == 0:
            photo_region_kp_desc[key] = ([], None, 0, 0)
            continue
        # Slice from already-equalized grayscale (no redundant CLAHE)
        pg = photo_gray_eq[int(oh * ry0f):int(oh * ry1f),
                           int(ow * rx0f):int(ow * rx1f)]
        kp_r, desc_r = sift.detectAndCompute(pg, None)
        photo_region_kp_desc[key] = (kp_r, desc_r, rh, rw)

    # --- Supabase query ---
    dex_num = pokemon_match.get('dex', 0) if pokemon_match else 0
    card_num_val = card_id.get('card_number', '')
    ocr_name_raw = ocr_fields.get('name', '')

    # Build clean search name
    all_pokemon_names = (pokemon_match.get('all_pokemon_names', [])
                         if pokemon_match else [])
    if all_pokemon_names:
        ocr_name_clean = '&'.join(all_pokemon_names)
        print(f"  [name-clean] extracted Pokemon names: {ocr_name_clean}")
    elif pokemon_match and pokemon_match.get('matched_name'):
        ocr_name_clean = pokemon_match['matched_name']
        print(f"  [name-clean] using matched name: {ocr_name_clean}")
    else:
        ocr_name_clean = ocr_name_raw.strip()

    # Strip leading zeros for DB queries
    card_num_stripped = card_num_val.lstrip('0') if card_num_val else ''
    card_num_variants = [card_num_val]
    if card_num_stripped and card_num_stripped != card_num_val:
        card_num_variants.append(card_num_stripped)

    # Detect secret rare
    set_size_str = card_id.get('set_size', '')
    is_secret_rare = False
    if card_num_val and set_size_str:
        try:
            is_secret_rare = int(card_num_val) > int(set_size_str)
        except ValueError:
            pass
    if is_secret_rare:
        print(f"  [query] Secret rare detected: {card_num_val}/{set_size_str}")

    def _query_card_number(filters, cn_key='card_number'):
        """Try query with each card_number variant."""
        for cn in card_num_variants:
            result = supabase_query({**filters, cn_key: cn})
            if result:
                return result
        return []

    query_results = []

    # 1) Multi-pokemon (&) cards: name pattern + card number, then name pattern alone
    if len(all_pokemon_names) > 1:
        name_pattern = f'*{all_pokemon_names[0]}*{all_pokemon_names[1]}*'
        if card_num_val:
            query_results = _query_card_number({'name': ('ilike', name_pattern)})
        if not query_results:
            query_results = supabase_query({'name': ('ilike', name_pattern)})

    # 2) Best identifier + card number (pokedex is more reliable than OCR name)
    if not query_results and card_num_val:
        if dex_num and not is_secret_rare:
            query_results = _query_card_number({'pokedex': dex_num})
        # Only try name+number if pokedex wasn't available or didn't match
        if not query_results and ocr_name_clean:
            query_results = _query_card_number(
                {'name': ('ilike', f'*{ocr_name_clean}*')})

    # 3) Broad fallbacks (no card number) — only if we have no results yet
    #    Skip name-only if name+number already failed (removing the number
    #    just adds wrong cards). Pokedex-only is still useful since it's
    #    a different identifier axis than name.
    if not query_results and dex_num:
        query_results = supabase_query({'pokedex': dex_num})

    # 4) Last resort: card number alone
    if not query_results and card_num_val:
        query_results = _query_card_number({})

    print(f"  DB candidates: {len(query_results)}")

    # Download reference images in parallel
    candidates_to_verify = query_results[:20]
    ref_images = {}
    if candidates_to_verify:
        def _download(row):
            url = row.get('image_url', '') or row.get('imageUrl', '') or ''
            if url:
                return row, download_image(url)
            return row, None

        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(_download, row) for row in candidates_to_verify]
            for f in as_completed(futures):
                row, img = f.result()
                if img is not None:
                    ref_images[id(row)] = img

    # --- SIFT verify with parallel matching + early exit ---
    EARLY_EXIT_THRESHOLD = 0.25

    def _sift_score_one(row):
        """Score a single candidate against pre-computed photo keypoints."""
        ref_img = ref_images.get(id(row))
        if ref_img is None:
            return row, 0, 0, 0.0, 0, 0, 0.0

        ref_g = (cv2.cvtColor(ref_img, cv2.COLOR_RGB2GRAY)
                 if ref_img.ndim == 3 else ref_img)

        # Full-image SIFT (reuses photo_kp, photo_desc)
        b_good, b_total, b_ratio = sift_match_score(
            photo_kp, photo_desc, ref_g, sift, clahe, photo_gray_for_sift.shape)

        # Region SIFT (reuses photo_region_kp_desc, same sift engine)
        ref_g_resized = cv2.resize(ref_g, (ow, oh), interpolation=cv2.INTER_AREA)
        r_good, r_total, r_ratio = region_sift_score(
            photo_region_kp_desc, ref_g_resized, oh, ow,
            sift, clahe, region_specs)

        return row, b_good, b_total, b_ratio, r_good, r_total, r_ratio

    candidates = []
    best_match = None

    # Submit all SIFT jobs in parallel
    with ThreadPoolExecutor(max_workers=4) as pool:
        future_to_row = {pool.submit(_sift_score_one, row): row
                         for row in candidates_to_verify}

        for i, future in enumerate(as_completed(future_to_row)):
            row, b_good, b_total, b_ratio, r_good, r_total, r_ratio = future.result()

            card_name = row.get('name', '') or ''
            row_card_num = row.get('card_number', '') or ''
            row_set = row.get('set_name', '') or ''
            row_rarity = row.get('rarity', '') or ''
            image_url = (row.get('image_url', '') or row.get('imageUrl', '') or '')
            combined = 0.6 * b_ratio + 0.4 * r_ratio

            c = {'db_row': row, 'name': card_name, 'card_number': row_card_num,
                 'set': row_set, 'rarity': row_rarity, 'image_url': image_url,
                 'blob_good': b_good, 'blob_total': b_total, 'blob_ratio': b_ratio,
                 'region_good': r_good, 'region_total': r_total,
                 'region_ratio': r_ratio, 'combined_score': combined}

            print(f"    [{i+1}] {card_name[:25]:<25} blob={b_ratio*100:.1f}% "
                  f"region={r_ratio*100:.1f}% combined={combined*100:.1f}%")

            candidates.append(c)

            # Early exit on high-confidence match
            if combined >= EARLY_EXIT_THRESHOLD:
                print(f"    ★ Early exit — high confidence match")
                best_match = c
                # Cancel remaining futures
                for f in future_to_row:
                    f.cancel()
                break

    if not best_match and candidates:
        candidates.sort(key=lambda c: c['combined_score'], reverse=True)
        best_match = candidates[0] if candidates[0]['combined_score'] > 0 else None

    # Store reference image for visualization
    if best_match:
        ref_img_for_viz = ref_images.get(id(best_match['db_row']))
        if ref_img_for_viz is not None:
            best_match['ref_img'] = ref_img_for_viz

    if best_match:
        print(f"  ★ MATCH: {best_match['name']} ({best_match['card_number']}) "
              f"— {best_match['combined_score']*100:.1f}%")
    else:
        print(f"  No match")

    # --- Build result ---
    result = {
        'blob_index': blob_index,
        'is_card': is_card,
        'ratio': ratio,
        'flipped': card_flipped,
        'ocr_detections': {'all_text': detections, **ocr_fields},
        'language': 'ja' if is_japanese else 'en',
        'ocr_engine': ocr_engine_used,
        'pokemon_match': pokemon_match,
        'card_id': card_id,
        'best_match': best_match,
        'candidates': len(candidates),
        'ocr_img': ocr_img,
    }

    # Prefer OCR-detected rarity over DB rarity
    ocr_rarity = card_id.get('rarity_code', '')
    display_rarity = ocr_rarity if ocr_rarity else (
        best_match.get('rarity', '') if best_match else '')

    if best_match:
        result['summary'] = (f"{best_match['name']} | {best_match['card_number']} | "
                             f"{best_match.get('set','')} | {display_rarity} | "
                             f"SIFT {best_match['combined_score']*100:.1f}%")
    elif pokemon_match:
        rarity_part = f" | {ocr_rarity}" if ocr_rarity else ""
        result['summary'] = (f"{pokemon_match.get('en','?')} "
                             f"({pokemon_match.get('matched_name','?')}) | "
                             f"Card# {card_id.get('card_number','?')}{rarity_part} | "
                             f"No DB match")
    else:
        name_raw = ocr_fields.get('name', '?')
        rarity_part = f" | {ocr_rarity}" if ocr_rarity else ""
        result['summary'] = (f"OCR: {name_raw[:30]} | "
                             f"Card# {card_id.get('card_number','?')}{rarity_part} | "
                             f"No match")

    return result


def process_single_blob_api(blob, color_img, enhanced, clahe,
                             blob_index, engines, pokemon_db,
                             orient_data, card_flipped, yomitoku_lock=None):
    """API wrapper: returns JSON-serializable result (no numpy arrays)."""
    result = process_single_blob(
        blob, color_img, enhanced, clahe, None,
        blob_index, engines, pokemon_db,
        orient_data, card_flipped, yomitoku_lock
    )
    # Strip non-serializable numpy arrays
    result.pop('ocr_img', None)
    if result.get('best_match'):
        result['best_match'].pop('ref_img', None)
    if 'ocr_detections' in result:
        result['ocr_detections'].pop('all_text', None)
    return result


# ============================================================
# MAIN SCANNER
# ============================================================

def tradeon_scipy_scanner(image_path):
    """Multi-card scanner: detect all card blobs, process each one."""
    total_start = time.time()
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    # Initialize OCR engines ONCE (lazy — loaded on first use)
    engines = OCREngines()

    # Load pokemon DB ONCE
    pokemon_db = load_pokemon_db(image_path)
    print(f"Pokemon DB: {len(pokemon_db)} entries loaded")

    # Detect all blobs
    result = detect_all_blobs(image_path)
    if not result[0]:
        print("No card-like blobs found.")
        return []

    blobs, raw_img, color_img, enhanced = result

    # Filter to card-like blobs
    card_blobs = [b for b in blobs if b['is_card_like']]
    if not card_blobs:
        print("No blobs with card-like aspect ratio. Using all blobs.")
        card_blobs = blobs[:5]

    print(f"\nProcessing {len(card_blobs)} card(s)...")

    # ── Phase 1: Orient all cards (fast corner-crop detection) ──
    print(f"\n--- Phase 1: Orientation detection (corner-crop) ---")
    orient_data_list = []
    for i, blob in enumerate(card_blobs):
        od = orient_card(blob, color_img, enhanced, engines, i)
        orient_data_list.append(od)

    # ── Phase 2: Majority vote ──
    WEAK_THRESHOLD = 3
    total_signal = sum(od['orient_score'] for od in orient_data_list)
    majority_flip = total_signal < 0

    orientations = []
    n_overridden = 0
    for i, od in enumerate(orient_data_list):
        score = od['orient_score']
        if score >= WEAK_THRESHOLD:
            individual_flip = False
        elif score <= -WEAK_THRESHOLD:
            individual_flip = True
        else:
            individual_flip = score < 0

        # For multi-card scans: override weak decisions with majority
        if len(card_blobs) >= 3 and abs(score) < WEAK_THRESHOLD:
            final_flip = majority_flip
            if final_flip != individual_flip:
                n_overridden += 1
                print(f"  [majority] Card {i+1}: overriding "
                      f"{'flip' if individual_flip else 'keep'} → "
                      f"{'flip' if final_flip else 'keep'} (score={score})")
        else:
            final_flip = individual_flip
        orientations.append(final_flip)

    n_flip = sum(orientations)
    n_keep = len(orientations) - n_flip
    print(f"\n--- Phase 2: Majority vote ---")
    print(f"  Total signal: {total_signal:+d} → "
          f"majority={'FLIP' if majority_flip else 'KEEP 0°'}")
    print(f"  Result: {n_flip} flipped, {n_keep} kept, {n_overridden} overridden")

    # ── Phase 3: Process each card (parallel where possible) ──
    print(f"\n--- Phase 3: Card processing (parallel) ---")

    # Strategy: Yomitoku is not thread-safe, but we can overlap:
    #   - SIFT computation (CPU-bound, releases GIL in OpenCV)
    #   - Image downloads (I/O-bound)
    #   - EasyOCR for EN cards (separate from Yomitoku)
    # We use a thread pool but serialize Yomitoku with a lock.
    import threading
    yomitoku_lock = threading.Lock()

    def _process_card_thread_safe(args):
        i, blob = args
        # Wrap process_single_blob to serialize Yomitoku access
        return process_single_blob(
            blob, color_img, enhanced, clahe, image_path,
            blob_index=i, engines=engines, pokemon_db=pokemon_db,
            orient_data=orient_data_list[i], card_flipped=orientations[i],
            yomitoku_lock=yomitoku_lock
        )

    all_results = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(_process_card_thread_safe, (i, blob)): i
                   for i, blob in enumerate(card_blobs)}
        # Collect results in original order
        results_by_idx = {}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results_by_idx[idx] = future.result()
            except Exception as e:
                print(f"  [!] Card {idx+1} failed: {e}")
                results_by_idx[idx] = {'summary': f'Error: {e}'}
        all_results = [results_by_idx[i] for i in range(len(card_blobs))]

    elapsed = time.time() - total_start
    print(f"\nAll {len(card_blobs)} cards processed in {elapsed:.1f}s")

    # ============================================================
    # SUMMARY
    # ============================================================
    print(f"\n{'='*60}")
    print(f"SCAN COMPLETE — {len(all_results)} card(s) found")
    print(f"{'='*60}")
    for i, r in enumerate(all_results):
        print(f"  [{i+1}] {r.get('summary', 'No data')}")

    # Clean up large arrays (no image saving on server)
    for r in all_results:
        r.pop('ocr_img', None)
        if r.get('best_match') and 'ref_img' in r.get('best_match', {}):
            r['best_match'].pop('ref_img', None)

    return all_results
