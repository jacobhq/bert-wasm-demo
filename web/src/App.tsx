import * as wasm from "wonnx-embeddings-repro"

function App() {
  async function run() {
      const result = await wasm.embed()
      console.log(result)
  }

  return (
    <>
      <button onClick={() => run()}>Run wasm</button>
    </>
  )
}

export default App