package auth_client

import (
	"bytes"
	"crypto/ecdsa"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math/rand"
	"net/http"
	"time"

	"github.com/ethereum/go-ethereum/crypto"
)

// AuthenticatedClient is a client that can send authenticated requests.
type AuthenticatedClient struct {
	privateKey *ecdsa.PrivateKey
	client     *http.Client
}

// New creates a new AuthenticatedClient.
func New(privateKeyHex string) (*AuthenticatedClient, error) {
	privateKey, err := crypto.HexToECDSA(privateKeyHex)
	if err != nil {
		return nil, fmt.Errorf("invalid private key: %w", err)
	}

	return &AuthenticatedClient{
		privateKey: privateKey,
		client:     &http.Client{},
	}, nil
}

// SendRequest sends an authenticated request to the specified URL.
func (c *AuthenticatedClient) SendRequest(method, url string, body []byte) (*http.Response, error) {
	timestamp := fmt.Sprintf("%d", time.Now().Unix())
	nonce := fmt.Sprintf("%x", rand.Int63())
	bodyHash := sha256.Sum256(body)

	messageStr := fmt.Sprintf("%x.%s.%s", bodyHash, timestamp, nonce)
	messageHash := crypto.Keccak256([]byte(fmt.Sprintf("\x19Ethereum Signed Message:\n%d%s", len(messageStr), messageStr)))

	signature, err := crypto.Sign(messageHash, c.privateKey)
	if err != nil {
		return nil, fmt.Errorf("failed to sign message: %w", err)
	}
	signature[64] += 27 // for EIP-155 compatibility

	address := crypto.PubkeyToAddress(c.privateKey.PublicKey).Hex()

	req, err := http.NewRequest(method, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Address", address)
	req.Header.Set("X-Timestamp", timestamp)
	req.Header.Set("X-Nonce", nonce)
	req.Header.Set("X-Signature", hex.EncodeToString(signature))

	return c.client.Do(req)
}
