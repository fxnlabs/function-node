package main

import (
	"net/http"
	"time"

	"github.com/fxnlabs/function-node/internal/auth"
	"github.com/fxnlabs/function-node/internal/challenge"
	"github.com/fxnlabs/function-node/internal/config"
	"github.com/fxnlabs/function-node/internal/logger"
	"github.com/fxnlabs/function-node/internal/openai"
	"github.com/fxnlabs/function-node/internal/registry"
	"go.uber.org/zap"
)

const configDir = "config.yaml"
const modelBackendDir = "backend.yaml"

func main() {
	cfg, err := config.LoadConfig(configDir)
	if err != nil {
		panic(err)
	}
	zapLogger, err := logger.New(cfg.Logger.Verbosity)
	if err != nil {
		panic(err)
	}
	log := zapLogger.Named("node")

	backendConfig, err := config.LoadBackendConfig(modelBackendDir)
	if err != nil {
		log.Fatal("failed to load backend config", zap.Error(err))
	}

	// Use for verifying signatures and preventing replay attacks.
	nonceCache := auth.NewNonceCache(5*time.Minute, 1*time.Minute)

	gatewayRegistry := registry.NewGatewayRegistry(cfg.Registry.Gateway.PollInterval)
	schedulerRegistry := registry.NewSchedulerRegistry(cfg.Registry.Scheduler.PollInterval)

	challengeHandler := challenge.ChallengeHandler(log)
	http.Handle("/challenge", auth.AuthMiddleware(challengeHandler, log, nonceCache, schedulerRegistry))

	oaiHandler := openai.NewOAIHandler(backendConfig, log)
	http.Handle("/v1/chat/completions", auth.AuthMiddleware(oaiHandler, log, nonceCache, gatewayRegistry))
	http.Handle("/v1/completions", auth.AuthMiddleware(oaiHandler, log, nonceCache, gatewayRegistry))
	http.Handle("/v1/embeddings", auth.AuthMiddleware(oaiHandler, log, nonceCache, gatewayRegistry))

	log.Info("Starting server on :8080")
	if err := http.ListenAndServe(":8080", nil); err != nil {
		log.Fatal("failed to start server", zap.Error(err))
	}
}
