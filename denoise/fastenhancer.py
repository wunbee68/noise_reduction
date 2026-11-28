import numpy as np


class FastEnhancer:
    def __init__(self, weights, threads=1):
        import onnxruntime as ort

        ort.disable_telemetry_events()
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(weights), sess_options=options, providers=["CPUExecutionProvider"]
        )
        waveform, *states = self.session.get_inputs()
        self.hop = waveform.shape[-1]
        self.delay = states[0].shape[-1]
        self.input_names = [waveform.name, *(state.name for state in states)]
        self.initial_state = [np.zeros(state.shape, dtype=np.float32) for state in states]

    def __call__(self, waveform):
        waveform = np.asarray(waveform, dtype=np.float32)
        length = len(waveform)
        # Flush the model's STFT overlap-add and round to a complete block.
        padding = self.delay + (-(length + self.delay)) % self.hop
        frames = np.pad(waveform, (0, padding)).reshape(-1, 1, self.hop)
        output = np.empty_like(frames)
        state = self.initial_state
        for index, frame in enumerate(frames):
            block, *state = self.session.run(None, dict(zip(self.input_names, [frame, *state])))
            output[index] = block
        return output.reshape(-1)[self.delay:self.delay + length]
