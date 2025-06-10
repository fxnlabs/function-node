package config

import (
	"os"
	"time"

	"gopkg.in/yaml.v3"
)

type Config struct {
	Node struct {
		KeystorePath string `yaml:"keystorePath"`
	} `yaml:"node"`
	Logger struct {
		Verbosity string `yaml:"verbosity"`
	} `yaml:"logger"`
	Registry struct {
		Gateway struct {
			PollInterval time.Duration `yaml:"pollInterval"`
		} `yaml:"gateway"`
		Scheduler struct {
			PollInterval time.Duration `yaml:"pollInterval"`
		} `yaml:"scheduler"`
		Node struct {
			PollInterval time.Duration `yaml:"pollInterval"`
		} `yaml:"node"`
	} `yaml:"registry"`
	RpcProvider string `yaml:"rpcProvider"`
}

func LoadConfig(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var config Config
	err = yaml.Unmarshal(data, &config)
	if err != nil {
		return nil, err
	}

	return &config, nil
}
