import os
import sys
import time
import hashlib
import secrets
import requests
from ecdsa import SigningKey, SECP256k1
from eth_account import Account

def main():
    # Get private key from environment variable
    private_key_hex = os.environ.get("PRIVATE_KEY")
    if not private_key_hex:
        print("Error: PRIVATE_KEY environment variable is not set.")
        sys.exit(1)

    # Get endpoint and request body from command line arguments
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <endpoint> <request_body>")
        sys.exit(1)

    endpoint = sys.argv[1]
    request_body = sys.argv[2]
    url = f"http://localhost:8080{endpoint}"

    # Generate required headers
    timestamp = str(int(time.time()))
    nonce = secrets.token_hex(16)
    body_hash = hashlib.sha256(request_body.encode('utf-8')).hexdigest()
    message_to_sign = f"{body_hash}.{timestamp}.{nonce}"

    # Sign the message
    private_key = SigningKey.from_string(bytes.fromhex(private_key_hex), curve=SECP256k1)
    signature = private_key.sign(message_to_sign.encode('utf-8'))

    # Get the address
    acct = Account.from_key(private_key_hex)
    address = acct.address

    print(f"Sending request to {url}")
    print(f"Address: {address}")
    print(f"Timestamp: {timestamp}")
    print(f"Nonce: {nonce}")
    print(f"Signature: {signature.hex()}")
    print("")

    headers = {
        "Content-Type": "application/json",
        "X-Address": address,
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
