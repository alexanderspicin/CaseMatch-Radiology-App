from pwdlib import PasswordHash

password_hasher = PasswordHash.recommended()

def get_password_hash(password):
    return password_hasher.hash(password)


def verify_password(password, password_hash):
    return password_hasher.verify(password, password_hash)
