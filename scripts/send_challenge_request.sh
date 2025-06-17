#!/bin/bash

# A script to send authenticated challenge requests to the function node.
#
# Usage:
# ./send_challenge_request.sh <request_body>
#
# Example:
# ./send_challenge_request.sh '{"type": "IDENTITY", "payload": {"nonce": "some_nonce"}}'
#
# Requires the PRIVATE_KEY environment variable to be set with your hex-encoded private key.
# The scheduler test key is available in scripts/scheduler_test_key.json
# You can export it like this:
# export PRIVATE_KEY=$(jq -r .private_key scripts/scheduler_test_key.json)

set -e

if [ -z "$PRIVATE_KEY" ]; then
  echo "Error: PRIVATE_KEY environment variable is not set."
  exit 1
fi

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <request_body>"
    exit 1
fi

REQUEST_BODY=$1
ENDPOINT="/challenge"
URL="http://localhost:8080$ENDPOINT"

# Generate required headers
TIMESTAMP=$(date +%s)
NONCE=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 32 | head -n 1)
BODY_HASH=$(echo -n "$REQUEST_BODY" | openssl dgst -sha256 -binary | xxd -p -c 256)
MESSAGE_TO_SIGN="$BODY_HASH.$TIMESTAMP.$NONCE"

# Sign the message
# Note: This requires a specific version of openssl that supports secp256k1.
# The following command creates a signature and then converts it to the correct format.
SIGNATURE=$(echo -n "$MESSAGE_TO_SIGN" | openssl dgst -sha256 -sign <(echo -n "302e0201010420$PRIVATE_KEY" | xxd -r -p) | xxd -p -c 256)

# Extract r and s from the signature and concatenate them.
R=$(echo $SIGNATURE | cut -c 9-72)
S=$(echo $SIGNATURE | cut -c 77-140)
# some S values are 63 characters, some are 64.
if [ ${#S} -eq 63 ]; then
  S="0$S"
fi
SIGNATURE_RS="$R$S"

# Get the public key from the private key
PUBLIC_KEY=$(openssl ec -inform PEM -text -noout -in <(echo -n "302e0201010420$PRIVATE_KEY" | xxd -r -p) 2>/dev/null | grep pub -A 5 | tail -n +2 | tr -d '[:space:]:' | sed 's/^04//')

echo "Sending request to $URL"
echo "Public Key: $PUBLIC_KEY"
echo "Timestamp: $TIMESTAMP"
echo "Nonce: $NONCE"
echo "Signature: $SIGNATURE_RS"
echo ""

curl -X POST "$URL" \
-H "Content-Type: application/json" \
-H "X-Public-Key: $PUBLIC_KEY" \
-H "X-Timestamp: $TIMESTAMP" \
-H "X-Nonce: $NONCE" \
-H "X-Signature: $SIGNATURE_RS" \
-d "$REQUEST_BODY"

echo ""
