package config

import (
	"net/http"
	"os"

	"go.uber.org/zap"
	"gopkg.in/yaml.v3"
)

type ModelBackend struct {
	BackendProvider string `yaml:"backendProvider"`
	URL             string `yaml:"url,omitempty"`
	FxnID           string `yaml:"fxnId"`
	APIKey          string `yaml:"apiKey,omitempty"`
	BearerToken     string `yaml:"bearerToken,omitempty"`
	ModelNameAlias  string `yaml:"modelNameAlias"`
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
