""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.
'''
_pad        = '_'
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"


# Export all symbols:
#symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
#symbols = symbols = [' ', 'I', 'N', 'U', 'a', 'b', 'by', 'ch', 'cl', 'd', 'dy', 'e', 'f', 'g', 'gy', 'h', 'hy', 'i', 'j', 'k', 'ky', 'm', 'my', 'n', 'ny', 'o', 'p', 'pau', 'py', 'r', 'ry', 's', 'sh', 'sil', 't', 'ts', 'ty', 'u', 'v', 'w', 'y', 'z']

# 1
#symbols =[' ','sil','S', 'rr', 'a', 'u', 'i', 'N', 'd', 'h', 'e', 'g', 'b', 'n', 'tS', 'p', 't', 'j', 'z', 'w', 'o', 'ts', 'k', 's', 'dZ', 'm']
# 2 jaconv
#symbols =[' ', 'b', 'u::', '', 'i::', 'd', 'z', 'ch', 'gy', 'i:', 'ry', 'r', 'i', 'y', 'ny', 'o', 'o::', 'py', 'u:', 'w', 'e:', 'by', 'a', 'j', 's', 'n', 'h', 't', 'p', 'a:', 'm', 'hy', 'f', 'k', 'my', 'N', 'sil', 'u', 'ky', 'o:', 'sh', 'g', 'e', 'q', 'ts']

# bird00
symbols = ['nfo', 'ala', 'alc', 'aob', 'aut', 'bba', 'bbb', 'bbc', 'baw', 'ngs', 'ngn', 'ae', 'ah', 'ai', 'ay', 'ad', 'af', 'ag', 'bq', 'br', 'ch', 'aj', 'ap', 'ac', 'ak', 'bu', 'bz', 'ca', 'cb', 'bp', 'bi', 'bl', 'be', 'cc', 'bx', 'cd', 'ce', 'cg', 'ci', 'ba', 'bc', 'bd', 'bf', 'bg', 'bj', 'bm', 'as', 'dm', 'fb', 'cr', 'fq', 'db', 'ec', 'cj', 'en', 'ds', 'he', 'cl', 'hg', 'au', 'am', 'dn', 'ao', 'io', 'an', 'aq', 'ar', 'dl', 'dp', 'dg', 'hb', 'dh', 'cn', 'di', 'co', 'cm', 'da', 'dd', 'em', 'fg', 'hh', 'ck', 'aa', 'df', 'fc', 'el', 'hd', 'ip', 'iq', 'ja', 'fe', 'fa', 'ic', 'cp', 'ea', 'jo', 'dr', 'ej', 'jn', 'eh', 'ed', 'hc', 'hj', 'ib', 'ab', 'eg', 'ha', 'hi', 'eq', 'hl', 'ih', 'jm', 'jh', 'ir', 'jc', 'je','nb', 'nw', 'no', 'nr', 'nl', 'jt', 'nt', 'ou', 'ra', 'lo', 'nv', 're', 'rh', 'jy', 'rn', 'kd', 'nn', 'na', 'nd', 'nc', 'nh', 'nj', 'ne', ' ', 'SIL']

# bird01_cath
# symbols = [' ', 'bba', 'aut', 'bbc', 'ngs', 'ngn', 'ala', 'nfo', 'bbb', 'baw', 'alc', 'aob', 'SIL']

# Special symbol ids
SPACE_ID = symbols.index(" ")
