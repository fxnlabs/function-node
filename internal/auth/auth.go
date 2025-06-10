package auth

import (
	"bytes"
	"crypto/ecdsa"
	"crypto/sha256"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"

	"github.com/ethereum/go-ethereum/crypto"
	"go.uber.org/zap"

	"github.com/fxnlabs/function-node/internal/registry"
)

type NonceCache struct {
	mu      sync.RWMutex
	nonces  map[string]time.Time
	ttl     time.Duration
	cleanup time.Duration
}

func NewNonceCache(ttl, cleanup time.Duration) *NonceCache {
	cache := &NonceCache{
		nonces:  make(map[string]time.Time),
		ttl:     ttl,
		cleanup: cleanup,
	}
	go cache.startCleanup()
	return cache
}

func (c *NonceCache) startCleanup() {
	ticker := time.NewTicker(c.cleanup)
	defer ticker.Stop()
	for range ticker.C {
		c.mu.Lock()
		for nonce, timestamp := range c.nonces {
			if time.Since(timestamp) > c.ttl {
				delete(c.nonces, nonce)
			}
		}
		c.mu.Unlock()
	}
}

func (c *NonceCache) IsUsed(nonce string) bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	_, used := c.nonces[nonce]
	return used
}

func (c *NonceCache) Use(nonce string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.nonces[nonce] = time.Now()
}

// AuthenticateChallenge verifies a signature against a message and public key.
func AuthenticateChallenge(signature, message []byte, publicKey ecdsa.PublicKey) (bool, error) {
	sigPublicKey, err := crypto.SigToPub(message, signature)
	if err != nil {
		return false, err
	}

	return sigPublicKey == &publicKey, nil
}

func AuthMiddleware(next http.Handler, log *zap.Logger, nonceCache *NonceCache, reg *registry.CachedRegistry) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		publicKey := r.Header.Get("X-Public-Key")
		if publicKey == "" {
			log.Warn("missing X-Public-Key header")
			http.Error(w, "missing X-Public-Key header", http.StatusUnauthorized)
			return
		}

		if _, ok := reg.Get(publicKey); !ok {
			log.Warn("node not registered", zap.String("publicKey", publicKey))
			http.Error(w, "node not registered", http.StatusUnauthorized)
			return
		}

		signature := []byte(r.Header.Get("X-Signature"))
		if len(signature) == 0 {
			log.Warn("missing X-Signature header")
			http.Error(w, "missing X-Signature header", http.StatusUnauthorized)
			return
		}

		privateKey, err := crypto.HexToECDSA(publicKey)
		if err != nil {
			log.Error("failed to parse public key", zap.String("publicKey", publicKey), zap.Error(err))
			http.Error(w, "invalid public key", http.StatusBadRequest)
			return
		}

		timestampStr := r.Header.Get("X-Timestamp")
		if timestampStr == "" {
			log.Warn("missing X-Timestamp header")
			http.Error(w, "missing X-Timestamp header", http.StatusUnauthorized)
			return
		}

		nonce := r.Header.Get("X-Nonce")
		if nonce == "" {
			log.Warn("missing X-Nonce header")
			http.Error(w, "missing X-Nonce header", http.StatusUnauthorized)
			return
		}

		if nonceCache.IsUsed(nonce) {
			log.Warn("nonce already used", zap.String("nonce", nonce))
			http.Error(w, "nonce already used", http.StatusUnauthorized)
			return
		}
		nonceCache.Use(nonce)

		bodyBytes, err := io.ReadAll(r.Body)
		if err != nil {
			log.Error("failed to read request body", zap.Error(err))
			http.Error(w, "internal server error", http.StatusInternalServerError)
			return
		}
		r.Body.Close()
		r.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))

		hash := sha256.Sum256(bodyBytes)
		message := []byte(fmt.Sprintf("%x.%s.%s", hash, timestampStr, nonce))

		valid, err := AuthenticateChallenge(signature, message, privateKey.PublicKey)
		if err != nil {
			log.Error("failed to authenticate request", zap.Error(err))
			http.Error(w, "internal server error", http.StatusInternalServerError)
			return
		}

		if !valid {
			log.Warn("invalid signature", zap.String("publicKey", publicKey))
			http.Error(w, "invalid signature", http.StatusUnauthorized)
			return
		}

		next.ServeHTTP(w, r)
	})
}
