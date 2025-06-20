package config

import (
	"os"
	"path/filepath"
	"time"

	"gopkg.in/yaml.v3"
)

func GetDefaultConfigHome() string {
	home, err := os.UserHomeDir()
	if err != nil {
		panic(err)
	}
	return filepath.Join(home, ".fxn")
}

type Config struct {
	Node struct {
		ListenPort int `yaml:"listenPort"`
	} `yaml:"node"`
	Logger struct {
		Verbosity string `yaml:"verbosity"`
	} `yaml:"logger"`
	Registry struct {
		RouterSmartContractAddress string `yaml:"routerSmartContractAddress"`
		Gateway                    struct {
			PollInterval time.Duration `yaml:"pollInterval"`
		} `yaml:"gateway"`
		Provider struct {
			PollInterval time.Duration `yaml:"pollInterval"`
		} `yaml:"provider"`
	} `yaml:"registry"`
	RpcProvider      string `yaml:"rpcProvider"`
	SchedulerAddress string `yaml:"schedulerAddress"`
	NonceCache       struct {
		TTL             time.Duration `yaml:"ttl"`
		CleanupInterval time.Duration `yaml:"cleanupInterval"`
	} `yaml:"nonceCache"`
	Proxy struct {
		MaxIdleConns    int           `yaml:"maxIdleConns"`
		IdleConnTimeout time.Duration `yaml:"idleConnTimeout"`
	} `yaml:"proxy"`
}

func LoadConfig(configPath string) (*Config, error) {
	data, err := os.ReadFile(configPath)
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
