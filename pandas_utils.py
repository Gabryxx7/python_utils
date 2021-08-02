from scipy.stats import kendalltau, pearsonr, spearmanr

import base64
import random
from Crypto.Cipher import AES
from Crypto.Util.strxor import strxor
from Crypto import Random
from itertools import cycle

def kendall_pval(x,y):
    return kendalltau(x,y)[1]

def kendall_corr(x,y):
    return kendalltau(x,y)[0]

def pearsonr_pval(x,y):
    return pearsonr(x,y)[1]

def pearsonr_corr(x,y):
    return pearsonr(x,y)[0]

def spearmanr_pval(x,y):
    return spearmanr(x,y)[1]

def spearmanr_corr(x,y):
    return spearmanr(x,y)[0]


def encrypt_df(df, key, cols_to_encrypt=None, method=0):
    if cols_to_encrypt is None:
        cols_to_encrypt = list(df.select_dtypes(['O']).columns)
    if method <= 0:
        df[cols_to_encrypt] = df[cols_to_encrypt].applymap(lambda x: encrypt_xor(x, key))
    else:
        df[cols_to_encrypt] = df[cols_to_encrypt].applymap(lambda x: encrypt_AES(x, key))
    return df

def decrypt_df(df, key, cols_to_decrypt=None, method=0):
    if cols_to_decrypt is None:
        cols_to_decrypt = list(df.select_dtypes(['O']).columns)
    
    if method <= 0:
        df[cols_to_decrypt] = df[cols_to_decrypt].applymap(lambda x: decrypt_xor(x, key))
    else:
        df[cols_to_decrypt] = df[cols_to_decrypt].applymap(lambda x: decrypt_AES(x, key))
    return df

def pad(s):
    s = str(s)
    bs = AES.block_size
    return s + (bs - len(s) % bs) * chr(bs - len(s) % bs)

def unpad(s):
    return s[:-ord(s[len(s)-1:])]

def xor(message, key):
    return ''.join(chr(ord(c)^ord(k)) for c,k in zip(message, cycle(key)))
    
def encrypt_xor(raw, key):
    xored = xor(raw, key).encode('utf-8')
    return str(base64.encodestring(xored).strip())[2:][:-1]

def decrypt_xor(enc, key):
    enc = base64.decodestring(enc.encode('utf-8')).decode('utf-8')
    xored = xor(enc, key)
    return xored

def encrypt_AES(raw, key):
    raw = pad(raw).encode('utf-8')
    iv = Random.new().read(AES.block_size)
    cipher = AES.new(key.encode('utf-8'), AES.MODE_CBC, iv)
    return base64.b64encode( iv + cipher.encrypt( raw ) ) 

def decrypt_AES(enc, key):
    enc = base64.b64decode(enc)
    iv = enc[:AES.block_size]
    cipher = AES.new(key.encode('utf-8'), AES.MODE_CBC, iv)
    return unpad(cipher.decrypt( enc[16:] ))

