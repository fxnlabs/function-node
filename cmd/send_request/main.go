package main

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	"github.com/fxnlabs/function-node/pkg/auth_client"
)

func main() {
	privateKeyHex := os.Getenv("PRIVATE_KEY")
	if privateKeyHex == "" {
		fmt.Println("Error: PRIVATE_KEY environment variable is not set.")
		os.Exit(1)
	}

	if len(os.Args) != 3 {
		fmt.Printf("Usage: %s <challenge|gateway> <request_body>\n", os.Args[0])
		os.Exit(1)
	}

	requestType := os.Args[1]
	requestBody := os.Args[2]
	var url string

	switch strings.ToLower(requestType) {
	case "challenge":
		url = "http://localhost:8080/challenge"
	case "gateway":
		url = "http://localhost:8080/gateway"
	default:
		fmt.Printf("Invalid request type: %s. Must be 'challenge' or 'gateway'.\n", requestType)
		os.Exit(1)
	}

	client, err := auth_client.New(privateKeyHex)
	if err != nil {
		fmt.Printf("Error creating client: %s\n", err)
		os.Exit(1)
	}

	resp, err := client.SendRequest("POST", url, []byte(requestBody))
	if err != nil {
		fmt.Printf("Error sending request: %s\n", err)
		os.Exit(1)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		fmt.Printf("request failed with status %d: %s\n", resp.StatusCode, string(bodyBytes))
		os.Exit(1)
	}

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		fmt.Printf("Error reading response body: %s\n", err)
		os.Exit(1)
	}

	fmt.Printf("Response: %s\n", string(respBody))
}
