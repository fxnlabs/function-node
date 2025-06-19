package main

import (
	"bytes"
	"crypto/ecdsa"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net/http"
	"os"
	"testing"
	"time"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/crypto"
	"github.com/fxnlabs/function-node/internal/config"
	"github.com/fxnlabs/function-node/mocks/contracts"
	mockethclient "github.com/fxnlabs/function-node/mocks/ethclient"
	"github.com/fxnlabs/function-node/mocks/registry"
	"github.com/fxnlabs/function-node/pkg/fxnclient"
	"github.com/phayes/freeport"
	"github.com/stretchr/testify/require"
)

func TestMainFunction(t *testing.T) {
	// Create a temporary keyfile
	privateKey, err := crypto.GenerateKey()
	require.NoError(t, err)
	keyFile, err := os.CreateTemp("", "keyfile.json")
	require.NoError(t, err)
	defer os.Remove(keyFile.Name())
	_, err = keyFile.WriteString(fmt.Sprintf(`{"address":"%s","private_key":"%s"}`,
		crypto.PubkeyToAddress(privateKey.PublicKey).Hex(),
		hex.EncodeToString(crypto.FromECDSA(privateKey))))
	require.NoError(t, err)
	keyFile.Close()

	// Get a free port for the server
	port, err := freeport.GetFreePort()
	require.NoError(t, err)

	// Create mock config
	cfg := &config.Config{}
	cfg.RpcProvider = "ws://localhost:8546"
	cfg.Node.ListenPort = port
	cfg.Node.Keyfile = keyFile.Name()
	cfg.Registry.RouterSmartContractAddress = "0x1F98431c8aD98523631AE4a59f267346ea31F984"
	cfg.NonceCache.TTL = 1 * time.Minute
	cfg.NonceCache.CleanupInterval = 1 * time.Minute
	cfg.Logger.Verbosity = "info"
	cfg.ModelBackendPath = "../../fixtures/tests/config/model_backend.yaml"

	// Create mock dependencies
	mockEthClient := new(mockethclient.MockEthClient)
	mockRouter := new(contracts.MockRouter)
	mockGatewayRegistry := new(registry.MockRegistry)
	mockSchedulerRegistry := new(registry.MockRegistry)
	mockProviderRegistry := new(registry.MockRegistry)

	// Setup mock expectations
	mockRouter.On("GetGatewayRegistry").Return(common.HexToAddress("0xGatewayRegistry"), nil)
	mockRouter.On("GetProviderRegistry").Return(common.HexToAddress("0xProviderRegistry"), nil)
	mockRouter.On("GetSchedulerRegistry").Return(common.HexToAddress("0xSchedulerRegistry"), nil)

	publicKey := privateKey.Public()
	publicKeyECDSA, ok := publicKey.(*ecdsa.PublicKey)
	require.True(t, ok)
	address := crypto.PubkeyToAddress(*publicKeyECDSA).Hex()
	mockSchedulerRegistry.On("Get", address).Return(struct{}{}, true)

	// Run the server in a goroutine
	go func() {
		err := run(cfg, mockEthClient, mockRouter, mockGatewayRegistry, mockSchedulerRegistry, mockProviderRegistry)
		require.NoError(t, err)
	}()

	// Give the server time to start
	time.Sleep(1 * time.Second)

	// Make a request to the /challenge endpoint
	body := []byte(`{"data":"test"}`)
	timestamp := fmt.Sprintf("%d", time.Now().Unix())
	nonce := "test-nonce"
	bodyHash := sha256.Sum256(body)
	messageStr := fmt.Sprintf("%x.%s.%s", bodyHash, timestamp, nonce)
	messageHash := fxnclient.EIP191Hash(messageStr)
	signature, err := crypto.Sign(messageHash, privateKey)
	require.NoError(t, err)
	signature[64] += 27 // for EIP-155 compatibility

	t.Logf("Signature: %s", hex.EncodeToString(signature))

	req, err := http.NewRequest("POST", fmt.Sprintf("http://localhost:%d/challenge", port), bytes.NewReader(body))
	require.NoError(t, err)
	req.Header.Set("X-Address", address)
	req.Header.Set("X-Signature", hex.EncodeToString(signature))
	req.Header.Set("X-Timestamp", timestamp)
	req.Header.Set("X-Nonce", nonce)

	t.Logf("Request Headers: %+v", req.Header)

	client := &http.Client{}
	resp, err := client.Do(req)
	require.NoError(t, err)
	defer resp.Body.Close()

	require.Equal(t, http.StatusOK, resp.StatusCode)
}
