package contracts

import (
	"errors"
	"strings"
	"testing"

	"github.com/ethereum/go-ethereum/accounts/abi"
	"github.com/ethereum/go-ethereum/common"
	"github.com/fxnlabs/function-node/fixtures"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"

	mocks "github.com/fxnlabs/function-node/mocks/ethclient"
)

func TestNewRouter(t *testing.T) {
	logger := zap.NewNop()
	mockClient := new(mocks.MockEthClient)
	contractAddress := common.HexToAddress("0x123")

	t.Run("success", func(t *testing.T) {
		router, err := NewRouter(mockClient, contractAddress, logger)
		require.NoError(t, err)
		assert.NotNil(t, router)
	})
}

func TestRouter_getAddress(t *testing.T) {
	logger := zap.NewNop()
	mockClient := new(mocks.MockEthClient)
	contractAddress := common.HexToAddress("0x123")
	router, err := NewRouter(mockClient, contractAddress, logger)
	require.NoError(t, err)

	methodName := "gatewayRegistry"
	expectedAddress := common.HexToAddress("0x456")

	// Pack the method call
	parsedABI, err := abi.JSON(strings.NewReader(fixtures.RouterABI))
	require.NoError(t, err)

	var packed []byte
	packed, err = parsedABI.Methods[methodName].Outputs.Pack(expectedAddress)
	require.NoError(t, err)

	t.Run("success", func(t *testing.T) {
		mockClient.On("CallContract", mock.Anything, mock.Anything, mock.Anything).Return(packed, nil).Once()
		addr, err := router.getAddress(methodName)
		require.NoError(t, err)
		assert.Equal(t, expectedAddress, addr)
		mockClient.AssertExpectations(t)
	})

	t.Run("call contract error", func(t *testing.T) {
		expectedErr := errors.New("call error")
		mockClient.On("CallContract", mock.Anything, mock.Anything, mock.Anything).Return(nil, expectedErr).Once()
		_, err := router.getAddress(methodName)
		assert.Error(t, err)
		assert.True(t, strings.Contains(err.Error(), expectedErr.Error()))
		mockClient.AssertExpectations(t)
	})

	t.Run("unpack error", func(t *testing.T) {
		mockClient.On("CallContract", mock.Anything, mock.Anything, mock.Anything).Return([]byte("invalid data"), nil).Once()
		_, err := router.getAddress(methodName)
		assert.Error(t, err)
		mockClient.AssertExpectations(t)
	})

	t.Run("pack error", func(t *testing.T) {
		_, err := router.getAddress("nonexistentMethod")
		assert.Error(t, err)
	})
}

func TestRouter_GetGatewayRegistryAddress(t *testing.T) {
	logger := zap.NewNop()
	mockClient := new(mocks.MockEthClient)
	contractAddress := common.HexToAddress("0x123")
	router, err := NewRouter(mockClient, contractAddress, logger)
	require.NoError(t, err)

	expectedAddress := common.HexToAddress("0xabc")

	// This is a bit of a hack to get the return value for the mock.
	// In a real scenario, the contract call would return the encoded address.
	var packed []byte
	packed, err = router.contractABI.Methods["gatewayRegistry"].Outputs.Pack(expectedAddress)
	require.NoError(t, err)

	mockClient.On("CallContract", mock.Anything, mock.Anything, mock.Anything).Return(packed, nil).Once()
	addr, err := router.GetGatewayRegistryAddress()
	require.NoError(t, err)
	assert.Equal(t, expectedAddress, addr)
	mockClient.AssertExpectations(t)
}

func TestRouter_GetProviderRegistryAddress(t *testing.T) {
	logger := zap.NewNop()
	mockClient := new(mocks.MockEthClient)
	contractAddress := common.HexToAddress("0x123")
	router, err := NewRouter(mockClient, contractAddress, logger)
	require.NoError(t, err)

	expectedAddress := common.HexToAddress("0xdef")

	// This is a bit of a hack to get the return value for the mock.
	// In a real scenario, the contract call would return the encoded address.
	var packed []byte
	packed, err = router.contractABI.Methods["providerRegistry"].Outputs.Pack(expectedAddress)
	require.NoError(t, err)

	mockClient.On("CallContract", mock.Anything, mock.Anything, mock.Anything).Return(packed, nil).Once()
	addr, err := router.GetProviderRegistryAddress()
	require.NoError(t, err)
	assert.Equal(t, expectedAddress, addr)
	mockClient.AssertExpectations(t)
}
