package registry

import (
	"errors"
	"math/big"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/ethereum/go-ethereum/accounts/abi"
	"github.com/ethereum/go-ethereum/common"
	"github.com/fxnlabs/function-node/internal/config"
	"github.com/fxnlabs/function-node/mocks/contracts"
	"github.com/fxnlabs/function-node/mocks/ethclient"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"go.uber.org/zap"
)

var providerABIPath = filepath.Join("../../", ProviderRegistryABIPath)

func TestNewProviderRegistry(t *testing.T) {
	cfg := &config.Config{}
	cfg.Registry.Provider.PollInterval = 1 * time.Second
	logger := zap.NewNop()

	t.Run("Success", func(t *testing.T) {
		mockClient := ethclient.NewMockEthClient(t)
		mockRouter := contracts.NewMockRouter(t)
		mockRouter.EXPECT().GetProviderRegistryAddress().Return(common.HexToAddress("0x123"), nil)

		// Mock the initial call to fetch the provider registry

		abiFileContent, err := os.ReadFile(providerABIPath)
		assert.NoError(t, err)
		parsedABI, err := abi.JSON(strings.NewReader(string(abiFileContent)))
		assert.NoError(t, err)
		expectedProviders := []Provider{}
		packedOutput, err := parsedABI.Methods["getActiveProvidersLive"].Outputs.Pack(expectedProviders)
		assert.NoError(t, err)
		mockClient.EXPECT().CallContract(mock.Anything, mock.Anything, mock.Anything).Return(packedOutput, nil)

		registry, err := NewProviderRegistry(mockClient, cfg, logger, mockRouter, providerABIPath)
		assert.NoError(t, err)
		assert.NotNil(t, registry)
	})

	t.Run("GetProviderRegistryAddressError", func(t *testing.T) {
		mockClient := ethclient.NewMockEthClient(t)
		mockRouter := contracts.NewMockRouter(t)
		mockRouter.EXPECT().GetProviderRegistryAddress().Return(common.Address{}, errors.New("router error"))
		_, err := NewProviderRegistry(mockClient, cfg, logger, mockRouter, providerABIPath)
		assert.Error(t, err)
	})
}

func TestFetchProviderRegistry(t *testing.T) {
	logger := zap.NewNop()
	contractAddress := common.HexToAddress("0x123")

	abiFileContent, err := os.ReadFile(providerABIPath)
	assert.NoError(t, err)
	parsedABI, err := abi.JSON(strings.NewReader(string(abiFileContent)))
	assert.NoError(t, err)

	t.Run("Success", func(t *testing.T) {
		mockClient := ethclient.NewMockEthClient(t)
		expectedProviders := []Provider{
			{
				Owner:        common.HexToAddress("0xabc"),
				Id:           []byte("provider1"),
				ModelId:      big.NewInt(1),
				RegisteredAt: big.NewInt(12345),
				Metadata:     "meta1",
				Paused:       false,
			},
		}
		packedOutput, err := parsedABI.Methods["getActiveProvidersLive"].Outputs.Pack(expectedProviders)
		assert.NoError(t, err)

		mockClient.EXPECT().CallContract(mock.Anything, mock.Anything, mock.Anything).Return(packedOutput, nil).Once()

		registryMap, err := fetchProviderRegistry(mockClient, contractAddress, parsedABI, logger)
		assert.NoError(t, err)
		assert.NotNil(t, registryMap)
		assert.Len(t, registryMap, 1)
		assert.Equal(t, expectedProviders[0], registryMap["provider1"])
	})

	t.Run("CallContractError", func(t *testing.T) {
		mockClient := ethclient.NewMockEthClient(t)
		mockClient.EXPECT().CallContract(mock.Anything, mock.Anything, mock.Anything).Return(nil, errors.New("contract call error")).Once()

		_, err := fetchProviderRegistry(mockClient, contractAddress, parsedABI, logger)
		assert.Error(t, err)
	})

	t.Run("UnpackError", func(t *testing.T) {
		mockClient := ethclient.NewMockEthClient(t)
		mockClient.EXPECT().CallContract(mock.Anything, mock.Anything, mock.Anything).Return([]byte("invalid data"), nil).Once()

		_, err := fetchProviderRegistry(mockClient, contractAddress, parsedABI, logger)
		assert.Error(t, err)
	})
}
