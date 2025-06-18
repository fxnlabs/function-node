package keys

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/crypto"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGenerateKeyFile(t *testing.T) {
	tempDir := t.TempDir()
	keyFilePath := filepath.Join(tempDir, "key.json")

	err := GenerateKeyFile(keyFilePath)
	require.NoError(t, err)

	assert.FileExists(t, keyFilePath)

	data, err := os.ReadFile(keyFilePath)
	require.NoError(t, err)

	var keyFile KeyFile
	err = json.Unmarshal(data, &keyFile)
	require.NoError(t, err)

	assert.NotEmpty(t, keyFile.PublicKey)
	assert.NotEmpty(t, keyFile.Address)
	assert.NotEmpty(t, keyFile.PrivateKey)

	// Verify that the keys are valid
	privateKey, err := crypto.HexToECDSA(keyFile.PrivateKey)
	require.NoError(t, err)

	publicKeyBytes := common.Hex2Bytes(keyFile.PublicKey)

	publicKey, err := crypto.UnmarshalPubkey(publicKeyBytes)
	require.NoError(t, err)

	assert.Equal(t, privateKey.PublicKey, *publicKey)
	assert.Equal(t, crypto.PubkeyToAddress(*publicKey).Hex(), keyFile.Address)
}

func TestLoadPrivateKey(t *testing.T) {
	tempDir := t.TempDir()
	keyFilePath := filepath.Join(tempDir, "key.json")

	// Generate a key file to use for testing
	err := GenerateKeyFile(keyFilePath)
	require.NoError(t, err)

	t.Run("successful load", func(t *testing.T) {
		privateKey, address, err := LoadPrivateKey(keyFilePath)
		require.NoError(t, err)
		assert.NotNil(t, privateKey)
		assert.NotEqual(t, common.Address{}, address)
	})

	t.Run("file not found", func(t *testing.T) {
		_, _, err := LoadPrivateKey(filepath.Join(tempDir, "nonexistent.json"))
		assert.Error(t, err)
	})

	t.Run("malformed json", func(t *testing.T) {
		malformedJSONPath := filepath.Join(tempDir, "malformed.json")
		err := os.WriteFile(malformedJSONPath, []byte("{"), 0600)
		require.NoError(t, err)

		_, _, err = LoadPrivateKey(malformedJSONPath)
		assert.Error(t, err)
	})

	t.Run("invalid private key", func(t *testing.T) {
		invalidKeyPath := filepath.Join(tempDir, "invalid_key.json")
		keyFile := KeyFile{
			PublicKey:  "04abc",
			Address:    "0x123",
			PrivateKey: "invalid",
		}
		data, err := json.Marshal(keyFile)
		require.NoError(t, err)
		err = os.WriteFile(invalidKeyPath, data, 0600)
		require.NoError(t, err)

		_, _, err = LoadPrivateKey(invalidKeyPath)
		assert.Error(t, err)
	})
}
