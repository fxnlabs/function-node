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

type ModelBackendConfig struct {
	Models map[string]string `yaml:"models"`
}

type ModelExtractor struct {
	Model string `json:"model"`
}

func LoadModelBackendConfig(path string) (*ModelBackendConfig, error) {
	data, err := os.ReadFile(path)
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

func (c *ModelBackendConfig) GetModelBackendURL(r *http.Request, log *zap.Logger) (string, error) {
	if r.Body == nil {
		log.Warn("request body is nil")
		return "", fmt.Errorf("request body is nil")
	}
	bodyBytes, err := io.ReadAll(r.Body)
	if err != nil {
		log.Warn("failed to read request body", zap.Error(err))
		return "", fmt.Errorf("failed to read request body: %w", err)
	}
	r.Body.Close()
	r.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))

	var modelExtractor ModelExtractor
	if err := json.Unmarshal(bodyBytes, &modelExtractor); err != nil {
		log.Warn("failed to unmarshal model from request body", zap.Error(err))
		return "", fmt.Errorf("failed to unmarshal model from request body: %w", err)
	}

	modelBackendURL, ok := c.Models[modelExtractor.Model]
	if !ok {
		log.Warn("model not found in model_backend config", zap.String("model", modelExtractor.Model))
		return "", fmt.Errorf("model not found in model_backend config: %s", modelExtractor.Model)
	}
	return modelBackendURL, nil
}
