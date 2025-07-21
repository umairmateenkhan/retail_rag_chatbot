import streamlit_authenticator as stauth

passwords = ['123', 'abc']
hashed_passwords = stauth.Hasher(passwords).generate()
print(hashed_passwords)
