// AudioWorklet processor for capturing PCM audio from microphone
class AudioProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0]
    if (input && input[0]) {
      // Convert Float32 to Int16
      const float32Data = input[0]
      const int16Data = new Int16Array(float32Data.length)
      for (let i = 0; i < float32Data.length; i++) {
        const s = Math.max(-1, Math.min(1, float32Data[i]))
        int16Data[i] = s < 0 ? s * 0x8000 : s * 0x7fff
      }
      // Send PCM data to main thread
      this.port.postMessage(int16Data.buffer)
    }
    return true
  }
}

registerProcessor('audio-processor', AudioProcessor)
