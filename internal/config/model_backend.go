package config

import (
	"net/http"
	"os"

	"go.uber.org/zap"
	"gopkg.in/yaml.v3"
)

type ModelBackend struct {
	BackendProvider string `yaml:"backend_provider"`
	URL             string `yaml:"url,omitempty"`
	FxnID           string `yaml:"fxn_id,omitempty"`
	APIKey          string `yaml:"api_key,omitempty"`
	BearerToken     string `yaml:"bearer_token,omitempty"`
}

func LoadModelBackend(configPath string) (*ModelBackend, error) {
	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil, err
	}

	var config ModelBackend
	err = yaml.Unmarshal(data, &config)
	if err != nil {
		return nil, err
	}

	return &config, nil
}
func (c *ModelBackend) GetModelBackend(r *http.Request, log *zap.Logger) (*ModelBackend, error) {
	return c, nil
}
