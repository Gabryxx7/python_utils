from pandas_utils import *
import pandas as pd


def test_encryption(filename, key='arandokey16bytes'):
    df = pd.read_csv(filename)
    enc_filename = "enc"+filename
    dec_filename = "dec"+filename
    print(f"df dtypes:\n {df.dtypes}\n")
    df_enc = encrypt_df(df, key)
    df_enc.to_csv(enc_filename)

    # Decrypt to double check
    df_enc = pd.read_csv(enc_filename)
    print(f"df ENC dtypes:\n {df_enc.dtypes}\n")
    df = decrypt_df(df_enc, key)
    df.to_csv(dec_filename)






if __name__ == "__main__":
    test_encryption("random_data.csv")
    test_encryption("random_data.csv")
