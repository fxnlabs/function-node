package config

import (
	"os"
	"time"

	"gopkg.in/yaml.v3"
)

type Config struct {
	Node struct {
		Keyfile string `yaml:"keyfile"`
	} `yaml:"node"`
	Logger struct {
		Verbosity string `yaml:"verbosity"`
	} `yaml:"logger"`
	Registry struct {
		RouterSmartContractAddress string `yaml:"routerSmartContractAddress"`
		Gateway                    struct {
			PollInterval time.Duration `yaml:"pollInterval"`
		} `yaml:"gateway"`
		Scheduler struct {
			SmartContractAddress string        `yaml:"smartContractAddress"`
			PollInterval         time.Duration `yaml:"pollInterval"`
		} `yaml:"scheduler"`
		Provider struct {
			PollInterval time.Duration `yaml:"pollInterval"`
		} `yaml:"provider"`
	} `yaml:"registry"`
	RpcProvider      string `yaml:"rpcProvider"`
	ModelBackendPath string `yaml:"modelBackendPath"`
	NonceCache       struct {
		TTL             time.Duration `yaml:"ttl"`
		CleanupInterval time.Duration `yaml:"cleanupInterval"`
	} `yaml:"nonceCache"`
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
