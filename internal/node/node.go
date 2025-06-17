package node

import (
	"crypto/ecdsa"
	"encoding/json"
	"os"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/crypto"
)

type KeyFile struct {
	PublicKey  string `json:"public_key"`
	Address    string `json:"address"`
	PrivateKey string `json:"private_key"`
}

func LoadPrivateKey(keyfile string) (*ecdsa.PrivateKey, common.Address, error) {
	data, err := os.ReadFile(keyfile)
	if err != nil {
		return nil, common.Address{}, err
	}

	var key KeyFile
	if err := json.Unmarshal(data, &key); err != nil {
		return nil, common.Address{}, err
	}

	privateKey, err := crypto.HexToECDSA(key.PrivateKey)
	if err != nil {
		return nil, common.Address{}, err
	}

	publicKey := privateKey.Public()
	publicKeyECDSA, ok := publicKey.(*ecdsa.PublicKey)
	if !ok {
		return nil, common.Address{}, err
	}

	address := crypto.PubkeyToAddress(*publicKeyECDSA)

	return privateKey, address, nil
}

func GenerateKeyFile(path string) error {
	privateKey, err := crypto.GenerateKey()
	if err != nil {
		return err
	}

	publicKey := privateKey.Public()
	publicKeyECDSA, ok := publicKey.(*ecdsa.PublicKey)
	if !ok {
		return err
	}

	address := crypto.PubkeyToAddress(*publicKeyECDSA).Hex()
	publicKeyBytes := crypto.FromECDSAPub(publicKeyECDSA)

	keyFile := KeyFile{
		PublicKey:  common.Bytes2Hex(publicKeyBytes),
		Address:    address,
		PrivateKey: common.Bytes2Hex(crypto.FromECDSA(privateKey)),
	}

	data, err := json.MarshalIndent(keyFile, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(path, data, 0600)
}
