## DeepSpeech Client

### Directory Structure
- [`deepspeech-0.9.3-models.pbmm`](https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm) and [`deepspeech-0.9.3-models.scorer`](https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer) files under `model/` directory.
- Input files under `data/input/` directory.
- Output files are generated to `data/output/` directory with the same filename from the input directory.

### Scoring using lm_phone.py
- To actually measure how much noise affects the performance of the model, we'll be using Phone Error Rate (PER).
- For comparing the output of a single sentence to the true sentence, you should use `input_setting = "word"`
- For comparing the output of an entire corpus, you should use `input_setting = "corpus"`
- Make sure to adjust the file and directory names accordingly for each, respectively. 
### Adding methods for noise
- I've updated `add_noise.py` to be more modular and allow us to add functions for quickly testing noisy output. 
If there's a specific type of noise you want to test, create a function and then call that function in the general `add_noise.py` function based on the if-else template. 
- The noise parameters are passed using the array `params`. Whatever parameters of the noise function you need from the main client file, you should pass through here. You should add parameters to the `NOISE_PARAMS` array in `client.py`. 