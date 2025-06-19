package main

import (
	"fmt"
	"net/http"

	"github.com/ethereum/go-ethereum/common"
	goethclient "github.com/ethereum/go-ethereum/ethclient"
	"github.com/fxnlabs/function-node/internal/auth"
	"github.com/fxnlabs/function-node/internal/challenge"
	"github.com/fxnlabs/function-node/internal/config"
	"github.com/fxnlabs/function-node/internal/contracts"
	"github.com/fxnlabs/function-node/internal/keys"
	"github.com/fxnlabs/function-node/internal/logger"
	"github.com/fxnlabs/function-node/internal/openai"
	"github.com/fxnlabs/function-node/internal/registry"
	"github.com/fxnlabs/function-node/pkg/ethclient"
	"go.uber.org/zap"
)

const configDir = "config.yaml"

func run(cfg *config.Config, ethClient ethclient.EthClient, router contracts.Router, gatewayRegistry registry.Registry, schedulerRegistry registry.Registry, providerRegistry registry.Registry) error {
	zapLogger, err := logger.New(cfg.Logger.Verbosity)
	if err != nil {
		return err
	}
	rootLogger := zapLogger.Named("node")
	modelBackendConfig, err := config.LoadModelBackendConfig(cfg.ModelBackendPath)
	if err != nil {
		rootLogger.Fatal("failed to load model_backend config", zap.Error(err))
	}

	privateKey, _, err := keys.LoadPrivateKey(cfg.Node.Keyfile)
	if err != nil {
		rootLogger.Fatal("failed to load private key", zap.Error(err))
	}

	// Use for verifying signatures and preventing replay attacks.
	nonceCache := auth.NewNonceCache(cfg.NonceCache.TTL, cfg.NonceCache.CleanupInterval)

	// Make providerRegistry available if needed by other parts, e.g., auth middleware
	// For now, it's initialized but not explicitly used further in this snippet.
	// If IsProviderRegistered is part of auth, it might use this providerRegistry.
	_ = providerRegistry // Placeholder to use providerRegistry, remove if it's passed to a consumer

	challengeHandler := challenge.ChallengeHandler(rootLogger, privateKey)
	// Assuming AuthMiddleware might need providerRegistry if it performs provider registration checks.
	// If not, schedulerRegistry might be the correct one for challenges.
	// Based on existing code, schedulerRegistry is used for /challenge
	http.Handle("/challenge", auth.AuthMiddleware(challengeHandler, rootLogger, nonceCache, schedulerRegistry))

	oaiProxyHandler := openai.NewOAIProxyHandler(modelBackendConfig, rootLogger)
	http.Handle("/v1/chat/completions", auth.AuthMiddleware(oaiProxyHandler, rootLogger, nonceCache, gatewayRegistry))
	http.Handle("/v1/completions", auth.AuthMiddleware(oaiProxyHandler, rootLogger, nonceCache, gatewayRegistry))
	http.Handle("/v1/embeddings", auth.AuthMiddleware(oaiProxyHandler, rootLogger, nonceCache, gatewayRegistry))

	modelsHandler := openai.NewModelsHandler(modelBackendConfig, rootLogger)
	http.Handle("/v1/models", auth.AuthMiddleware(modelsHandler, rootLogger, nonceCache, gatewayRegistry))

	addr := fmt.Sprintf(":%d", cfg.Node.ListenPort)
	rootLogger.Info("Starting server on", zap.String("address", addr))
	if err := http.ListenAndServe(addr, nil); err != nil {
		rootLogger.Fatal("failed to start server", zap.Error(err))
	}
	return nil
}

func main() {
	cfg, err := config.LoadConfig(configDir)
	if err != nil {
		panic(err)
	}
	zapLogger, err := logger.New(cfg.Logger.Verbosity)
	if err != nil {
		panic(err)
	}
	rootLogger := zapLogger.Named("node")

	// Initialize Ethereum client
	ethClient, err := goethclient.Dial(cfg.RpcProvider)
	if err != nil {
		rootLogger.Fatal("Failed to connect to Ethereum RPC provider", zap.String("provider", cfg.RpcProvider), zap.Error(err))
	}
	defer ethClient.Close()

	// Initialize router
	routerAddress := common.HexToAddress(cfg.Registry.RouterSmartContractAddress)
	router, err := contracts.NewRouter(ethClient, routerAddress, rootLogger, contracts.DefaultRouterABIPath)
	if err != nil {
		rootLogger.Fatal("failed to create router", zap.Error(err))
	}

	// Initialize registries
	gatewayRegistry, err := registry.NewGatewayRegistry(ethClient, cfg, rootLogger, router, registry.GatewayRegistryABIPath)
	if err != nil {
		rootLogger.Fatal("failed to initialize gateway registry", zap.Error(err))
	}

	schedulerRegistry, err := registry.NewSchedulerRegistry(ethClient, cfg, rootLogger)
	if err != nil {
		rootLogger.Fatal("failed to initialize scheduler registry", zap.Error(err))
	}

	providerRegistry, err := registry.NewProviderRegistry(ethClient, cfg, rootLogger, router, registry.ProviderRegistryABIPath)
	if err != nil {
		rootLogger.Fatal("failed to initialize provider registry", zap.Error(err))
	}

	if err := run(cfg, ethClient, router, gatewayRegistry, schedulerRegistry, providerRegistry); err != nil {
		rootLogger.Fatal("failed to run application", zap.Error(err))
	}
}
