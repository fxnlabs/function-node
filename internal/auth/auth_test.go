package auth

import (
	"bytes"
	"crypto/ecdsa"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/ethereum/go-ethereum/crypto"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"

	mocks "github.com/fxnlabs/function-node/mocks/registry"
)

func TestNonceCache(t *testing.T) {
	// Test with a short TTL for quick testing
	cache := NewNonceCache(100*time.Millisecond, 10*time.Millisecond)

	// Test case 1: Nonce not used
	assert.False(t, cache.IsUsed("nonce1"), "Nonce should not be used yet")

	// Test case 2: Use nonce
	cache.Use("nonce1")
	assert.True(t, cache.IsUsed("nonce1"), "Nonce should be marked as used")

	// Test case 3: Nonce expires
	time.Sleep(150 * time.Millisecond)
	assert.False(t, cache.IsUsed("nonce1"), "Nonce should have expired")

	// Test case 4: Cleanup removes expired nonces
	cache.Use("nonce2")
	time.Sleep(150 * time.Millisecond)
	cache.mu.Lock()
	_, exists := cache.nonces["nonce2"]
	cache.mu.Unlock()
	assert.False(t, exists, "Expired nonce should be removed by cleanup")
}

func TestAuthenticateChallenge(t *testing.T) {
	privateKey, err := crypto.GenerateKey()
	require.NoError(t, err)

	publicKey := privateKey.Public()
	publicKeyECDSA, ok := publicKey.(*ecdsa.PublicKey)
	require.True(t, ok)

	address := crypto.PubkeyToAddress(*publicKeyECDSA).Hex()
	message := []byte("test message")
	hash := crypto.Keccak256Hash(message)

	signature, err := crypto.Sign(hash.Bytes(), privateKey)
	require.NoError(t, err)

	// Test case 1: Valid signature
	valid, err := AuthenticateChallenge(signature, hash.Bytes(), address)
	assert.NoError(t, err)
	assert.True(t, valid, "Signature should be valid")

	// Test case 2: Invalid signature
	invalidSignature := []byte("invalid signature")
	valid, err = AuthenticateChallenge(invalidSignature, hash.Bytes(), address)
	assert.Error(t, err)
	assert.False(t, valid, "Signature should be invalid")

	// Test case 3: Wrong address
	valid, err = AuthenticateChallenge(signature, hash.Bytes(), "0x0000000000000000000000000000000000000000")
	assert.NoError(t, err)
	assert.False(t, valid, "Signature should be invalid for wrong address")
}

func TestAuthMiddleware(t *testing.T) {
	privateKey, err := crypto.GenerateKey()
	require.NoError(t, err)

	publicKey := privateKey.Public()
	publicKeyECDSA, ok := publicKey.(*ecdsa.PublicKey)
	require.True(t, ok)

	address := crypto.PubkeyToAddress(*publicKeyECDSA).Hex()

	log := zap.NewNop()
	nonceCache := NewNonceCache(1*time.Minute, 10*time.Second)
	mockReg := new(mocks.MockRegistry)

	// Setup mock expectations
	mockReg.On("Get", address).Return(struct{}{}, true)
	mockReg.On("Get", "0x0000000000000000000000000000000000000000").Return(nil, false)

	nextHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("OK"))
	})

	authHandler := AuthMiddleware(nextHandler, log, nonceCache, mockReg)

	t.Run("valid request", func(t *testing.T) {
		body := []byte(`{"hello":"world"}`)
		timestamp := fmt.Sprintf("%d", time.Now().Unix())
		nonce := "test-nonce-1"

		bodyHash := sha256.Sum256(body)
		messageStr := fmt.Sprintf("%x.%s.%s", bodyHash, timestamp, nonce)
		messageHash := crypto.Keccak256([]byte(fmt.Sprintf("\x19Ethereum Signed Message:\n%d%s", len(messageStr), messageStr)))

		signature, err := crypto.Sign(messageHash, privateKey)
		require.NoError(t, err)

		req := httptest.NewRequest("POST", "/", bytes.NewReader(body))
		req.Header.Set("X-Address", address)
		req.Header.Set("X-Signature", hex.EncodeToString(signature))
		req.Header.Set("X-Timestamp", timestamp)
		req.Header.Set("X-Nonce", nonce)

		rr := httptest.NewRecorder()
		authHandler.ServeHTTP(rr, req)

		assert.Equal(t, http.StatusOK, rr.Code)
		bodyBytes, err := io.ReadAll(rr.Body)
		require.NoError(t, err)
		assert.Equal(t, "OK", string(bodyBytes))
	})

	t.Run("missing X-Address", func(t *testing.T) {
		req := httptest.NewRequest("POST", "/", strings.NewReader(""))
		rr := httptest.NewRecorder()
		authHandler.ServeHTTP(rr, req)
		assert.Equal(t, http.StatusUnauthorized, rr.Code)
	})

	t.Run("node not registered", func(t *testing.T) {
		req := httptest.NewRequest("POST", "/", strings.NewReader(""))
		req.Header.Set("X-Address", "0x0000000000000000000000000000000000000000")
		rr := httptest.NewRecorder()
		authHandler.ServeHTTP(rr, req)
		assert.Equal(t, http.StatusUnauthorized, rr.Code)
	})

	t.Run("nonce already used", func(t *testing.T) {
		body := []byte(`{"hello":"world"}`)
		timestamp := fmt.Sprintf("%d", time.Now().Unix())
		nonce := "test-nonce-2"

		// Create a valid signature for this request
		bodyHash := sha256.Sum256(body)
		messageStr := fmt.Sprintf("%x.%s.%s", bodyHash, timestamp, nonce)
		messageHash := crypto.Keccak256([]byte(fmt.Sprintf("\x19Ethereum Signed Message:\n%d%s", len(messageStr), messageStr)))
		signature, err := crypto.Sign(messageHash, privateKey)
		require.NoError(t, err)

		// Use the nonce before the request
		nonceCache.Use(nonce)

		req := httptest.NewRequest("POST", "/", bytes.NewReader(body))
		req.Header.Set("X-Address", address)
		req.Header.Set("X-Signature", hex.EncodeToString(signature))
		req.Header.Set("X-Timestamp", timestamp)
		req.Header.Set("X-Nonce", nonce)

		rr := httptest.NewRecorder()
		authHandler.ServeHTTP(rr, req)

		assert.Equal(t, http.StatusUnauthorized, rr.Code)
	})

	t.Run("invalid signature", func(t *testing.T) {
		body := []byte(`{"hello":"world"}`)
		timestamp := fmt.Sprintf("%d", time.Now().Unix())
		nonce := "test-nonce-3"

		req := httptest.NewRequest("POST", "/", bytes.NewReader(body))
		req.Header.Set("X-Address", address)
		req.Header.Set("X-Signature", "0x123")
		req.Header.Set("X-Timestamp", timestamp)
		req.Header.Set("X-Nonce", nonce)

		rr := httptest.NewRecorder()
		authHandler.ServeHTTP(rr, req)

		assert.Equal(t, http.StatusBadRequest, rr.Code)
	})
}
