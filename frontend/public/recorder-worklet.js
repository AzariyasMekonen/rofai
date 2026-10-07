// Copies raw mic samples to the main thread; resampling and WAV encoding happen there.
class RecorderProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (channel) this.port.postMessage(channel.slice(0));
    return true;
  }
}

registerProcessor('recorder', RecorderProcessor);
