import { useState, useEffect } from 'react';
import { Model } from 'bert-wasm-demo';

export default function App() {
  const [model, setModel] = useState(null);
  const [inputText, setInputText] = useState('');
  const [embeddings, setEmbeddings] = useState(null);
  const [loading, setLoading] = useState(true);

  const modelURL = 'https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/refs%2Fpr%2F21/model.safetensors';
  const tokenizerURL = 'https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/refs%2Fpr%2F21/tokenizer.json';
  const configURL = 'https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/refs%2Fpr%2F21/config.json';

  useEffect(() => {
    const initializeModel = async () => {
      const fetchArrayBuffer = async (url) => {
        const response = await fetch(url);
        return new Uint8Array(await response.arrayBuffer());
      };

      const [weightsArrayU8, tokenizerArrayU8, configArrayU8] = await Promise.all([
        fetchArrayBuffer(modelURL),
        fetchArrayBuffer(tokenizerURL),
        fetchArrayBuffer(configURL)
      ]);

      const loadedModel = new Model(weightsArrayU8, tokenizerArrayU8, configArrayU8);
      setModel(loadedModel);
      setLoading(false);
    };

    initializeModel();
  }, [model]);

  const handleInputChange = (e) => {
    setInputText(e.target.value);
  };

  const getEmbeddings = async () => {
    if (!inputText) {
      alert('Please enter some text.');
      return;
    }

    console.log(inputText)
    const result = await model.get_embeddings({
      sentences: [inputText],
      normalize_embeddings: true
    });
    console.log(result)

    setEmbeddings(result.data[0]);
  };

  if (loading) {
    return <div>Loading model...</div>;
  }

  return (
    <div>
      <h1>Simple BERT Embeddings Example</h1>
      <textarea
        rows={4}
        cols={50}
        value={inputText}
        onChange={handleInputChange}
        placeholder="Enter text here..."
      />
      <br />
      <button onClick={getEmbeddings}>Get Embeddings</button>
      <br />
      {embeddings && (
        <pre>{JSON.stringify(embeddings, null, 2)}</pre>
      )}
    </div>
  );
}