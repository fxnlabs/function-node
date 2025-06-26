package config

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"

	"go.uber.org/zap"
	"gopkg.in/yaml.v3"
)

type AuthConfig struct {
	APIKey      string `yaml:"api_key,omitempty"`
	BearerToken string `yaml:"bearer_token,omitempty"`
}

type ModelBackend struct {
	URL  string      `yaml:"url"`
	Auth *AuthConfig `yaml:"auth,omitempty"`
}

type ModelBackendConfig struct {
	Models map[string]ModelBackend `yaml:"models"`
}

type ModelExtractor struct {
	Model string `json:"model"`
}

func LoadModelBackendConfig(configPath string) (*ModelBackendConfig, error) {
	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil, err
	}

	var config ModelBackendConfig
	err = yaml.Unmarshal(data, &config)
	if err != nil {
		return nil, err
	}

	return &config, nil
}

func (c *ModelBackendConfig) GetModelBackend(r *http.Request, log *zap.Logger) (*ModelBackend, error) {
	if r.Body == nil {
		log.Warn("request body is nil")
		return nil, fmt.Errorf("request body is nil")
	}
	bodyBytes, err := io.ReadAll(r.Body)
	if err != nil {
		log.Warn("failed to read request body", zap.Error(err))
		return nil, fmt.Errorf("failed to read request body: %w", err)
	}
	r.Body.Close()
	r.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))

	var modelExtractor ModelExtractor
	err = json.Unmarshal(bodyBytes, &modelExtractor)

	var modelName string
	if err != nil {
		log.Info("could not extract model from request body, using default", zap.Error(err))
		modelName = "default"
	} else {
		modelName = modelExtractor.Model
	}

	modelBackend, ok := c.Models[modelName]
	if !ok {
		if modelName != "default" {
			// fallback to default if model not found
			log.Info("model not found, falling back to default", zap.String("model", modelName))
			modelBackend, ok = c.Models["default"]
		}
		if !ok {
			log.Warn("model not found in model_backend config", zap.String("model", modelName))
			return nil, fmt.Errorf("model not found in model_backend config: %s", modelName)
		}
	}

	return &modelBackend, nil
}
