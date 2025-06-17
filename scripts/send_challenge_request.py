import os
import sys
import time
import hashlib
import secrets
import requests
from ecdsa import SigningKey, SECP256k1

def main():
    # Get private key from environment variable
    private_key_hex = os.environ.get("PRIVATE_KEY")
    if not private_key_hex:
        print("Error: PRIVATE_KEY environment variable is not set.")
        sys.exit(1)

    # Get request body from command line argument
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <request_body>")
        sys.exit(1)

    request_body = sys.argv[1]
    endpoint = "/challenge"
    url = f"http://localhost:8080{endpoint}"

    # Generate required headers
    timestamp = str(int(time.time()))
    nonce = secrets.token_hex(16)
    body_hash = hashlib.sha256(request_body.encode('utf-8')).hexdigest()
    message_to_sign = f"{body_hash}.{timestamp}.{nonce}"

    # Sign the message
    private_key = SigningKey.from_string(bytes.fromhex(private_key_hex), curve=SECP256k1)
    signature = private_key.sign(message_to_sign.encode('utf-8'))

    # Get the public key
    public_key = private_key.get_verifying_key().to_string("compressed").hex()

    print(f"Sending request to {url}")
    print(f"Public Key: {public_key}")
    print(f"Timestamp: {timestamp}")
    print(f"Nonce: {nonce}")
    print(f"Signature: {signature.hex()}")
    print("")

    headers = {
        "Content-Type": "application/json",
        "X-Public-Key": public_key,
        "X-Timestamp": timestamp,
        "X-Nonce": nonce,
        "X-Signature": signature.hex()
    }

    try:
        response = requests.post(url, headers=headers, data=request_body)
        print(f"Status Code: {response.status_code}")
        print("Response Body:")
        print(response.text)
    except requests.exceptions.RequestException as e:
        print(f"Error sending request: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
