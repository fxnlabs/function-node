package main

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	"github.com/fxnlabs/function-node/pkg/fxnclient"
)

func main() {
	privateKeyHex := os.Getenv("PRIVATE_KEY")
	if privateKeyHex == "" {
		fmt.Println("Error: PRIVATE_KEY environment variable is not set.")
		os.Exit(1)
	}

	if len(os.Args) != 3 {
		fmt.Printf("Usage: %s <endpoint> <request_body>\n\n", os.Args[0])
		fmt.Println("Example: go run cmd/send_request/main.go /v1/chat/completions '{\"model\":\"gpt-3.5-turbo\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}]}'")
		os.Exit(1)
	}

	endpoint := os.Args[1]
	if !strings.HasPrefix(endpoint, "/") {
		endpoint = "/" + endpoint
	}
	requestBody := os.Args[2]
	url := "http://localhost:8080" + endpoint

	httpClient := &http.Client{}
	client, err := fxnclient.NewFxnClient(privateKeyHex, httpClient)
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
