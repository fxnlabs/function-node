import os
import sys
import time
import hashlib
import secrets
import requests
from eth_account import Account
from eth_account.messages import encode_defunct

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
    message_to_sign_str = f"{body_hash}.{timestamp}.{nonce}"
    message_to_sign = encode_defunct(text=message_to_sign_str)

    # Sign the message
    acct = Account.from_key(private_key_hex)
    signed_message = acct.sign_message(message_to_sign)
    address = acct.address

    print(f"Sending request to {url}")
    print(f"Address: {address}")
    print(f"Timestamp: {timestamp}")
    print(f"Nonce: {nonce}")
    print(f"Signature: {signed_message.signature.hex()}")
    print("")

    headers = {
        "Content-Type": "application/json",
        "X-Address": address,
        "X-Timestamp": timestamp,
        "X-Nonce": nonce,
        "X-Signature": signed_message.signature.hex()
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
