package fxnclient

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

// FxnClient is a client that can send authenticated requests.
type FxnClient struct {
	privateKey *ecdsa.PrivateKey
	client     *http.Client
}

// NewFxnClient creates a new FxnClient.
func NewFxnClient(privateKeyHex string, client *http.Client) (*FxnClient, error) {
	privateKey, err := crypto.HexToECDSA(privateKeyHex)
	if err != nil {
		return nil, fmt.Errorf("invalid private key: %w", err)
	}

	return &FxnClient{
		privateKey: privateKey,
		client:     client,
	}, nil
}

// SendRequest sends an authenticated request to the specified URL.
func (c *FxnClient) SendRequest(method, url string, body []byte) (*http.Response, error) {
	req, err := http.NewRequest(method, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	timestamp := fmt.Sprintf("%d", time.Now().Unix())
	nonce := fmt.Sprintf("%x", rand.Int63())

	if err := c.SignRequest(req, body, timestamp, nonce); err != nil {
		return nil, err
	}

	return c.client.Do(req)
}

// SignRequest signs an http.Request with the client's private key.
func (c *FxnClient) SignRequest(req *http.Request, body []byte, timestamp, nonce string) error {
	bodyHash := sha256.Sum256(body)
	messageStr := fmt.Sprintf("%x.%s.%s", bodyHash, timestamp, nonce)
	messageHash := EIP191Hash(messageStr)

	signature, err := crypto.Sign(messageHash, c.privateKey)
	if err != nil {
		return fmt.Errorf("failed to sign message: %w", err)
	}
	signature[64] += 27 // for EIP-155 compatibility

	address := crypto.PubkeyToAddress(c.privateKey.PublicKey).Hex()

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Address", address)
	req.Header.Set("X-Timestamp", timestamp)
	req.Header.Set("X-Nonce", nonce)
	req.Header.Set("X-Signature", hex.EncodeToString(signature))
	return nil
}

// EIP191Hash computes the EIP-191 hash of a message.
func EIP191Hash(message string) []byte {
	return crypto.Keccak256([]byte(fmt.Sprintf("\x19Ethereum Signed Message:\n%d%s", len(message), message)))
}

// VerifySignature verifies a signature against a message and an address.
func VerifySignature(signature, message []byte, address string) (bool, error) {
	if len(signature) == 65 {
		// EIP-155 replay protection, v is 27 or 28
		if signature[64] == 27 || signature[64] == 28 {
			signature[64] -= 27
		}
	}

	sigPublicKey, err := crypto.SigToPub(message, signature)
	if err != nil {
		return false, err
	}

	recoveredAddress := crypto.PubkeyToAddress(*sigPublicKey).Hex()
	return recoveredAddress == address, nil
}
