package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/ledongthuc/pdf"
	"github.com/nguyenthenguyen/docx"
)

type Document struct {
	PageContent string            `json:"page_content"`
	Metadata    map[string]string `json:"metadata"`
}

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "Usage: processor <file1> <file2> ...")
		os.Exit(1)
	}

	var allText strings.Builder
	metadata := make(map[string]string)

	for _, filename := range os.Args[1:] {
		ext := strings.ToLower(filepath.Ext(filename))
		var text string
		var err error

		switch ext {
		case ".pdf":
			text, err = readPDF(filename)
		case ".docx":
			text, err = readDOCX(filename)
		case ".txt":
			text, err = readTXT(filename)
		default:
			fmt.Fprintf(os.Stderr, "Unsupported file type: %s\n", ext)
			os.Exit(1)
		}

		if err != nil {
			fmt.Fprintln(os.Stderr, "Error reading", filename, ":", err)
			os.Exit(1)
		}

		allText.WriteString(text)
		allText.WriteString("\n\n") // separate files with blank line

		// Track last processed file in metadata
		metadata["last_source"] = filename
	}

	// Wrap in a slice of documents (LangChain expects list-like behavior)
	docs := []Document{
		{
			PageContent: allText.String(),
			Metadata:    metadata,
		},
	}

	// Pretty-print JSON for readability
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	enc.Encode(docs)
}

func readPDF(path string) (string, error) {
	f, r, err := pdf.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()

	var sb strings.Builder
	totalPage := r.NumPage()
	for pageIndex := 1; pageIndex <= totalPage; pageIndex++ {
		p := r.Page(pageIndex)
		if p.V.IsNull() {
			continue
		}
		text, err := p.GetPlainText(nil)
		if err != nil {
			return "", err
		}
		sb.WriteString(text)
		sb.WriteString("\n")
	}
	return sb.String(), nil
}

func readDOCX(path string) (string, error) {
	r, err := docx.ReadDocxFile(path)
	if err != nil {
		return "", err
	}
	defer r.Close()

	return r.Editable().GetContent(), nil
}

func readTXT(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	return string(data), nil
}
