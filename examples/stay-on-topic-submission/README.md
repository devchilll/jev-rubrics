# Reproduction of "Stay on Topic with Classifier-Free Guidance"

## Setup

Either use `uv`:

```bash
uv sync
```

Or manually install the dependencies:

```console
pip install -r requirements.txt
```

## Reproduction script

We provide a singular `reproduce.sh` script to run all the reproduction commands
outlined below in a single sweep, end-to-end.

```console
bash reproduce.sh
```

## Section 3.1

The `run_3.1.py` script can be used to reproduce the results of section 3.1,
specifically the results shown in Table 5.

In particular, the script can run the Eleuther evaluation harness on
GPT-2-small, GPT-2-medium, GPT-2-large, GPT-2-XL, Pythia-160M, Pythia-410M,
Pythia-1B, Pythia-1.4B and Pythia-2.8B on ARC-c, ARC-e, BoolQ, HellaSwag, PiQA,
SciQ, TriviaQA, WinoGrande and LAMBADA (OpenAI).

We omit the evaluation of Llama models, since these require authorization
through a google form to use, and are therefore out of scope. Similarly, we omit
the evaluation of Pythia-6.9B and Pythia-12B, since these models are too large
to fit on our A10 hardware.

To run the script, simply use the following command:

```console
python run_3.1.py
```

The script will evaluate the models, and ultimately save the result for each of
the 9 tasks as a csv in the folder `results/table5/`. So for example it
will produce `results/table5/sciq_results.csv`, showing the performance on
the harness for the GPT2 and Pythia models with (gamma=1.5) and without
(gamma=1) classifier-free guidance on the SciQ task.

For time and computational reasons, we have pre-emptively run a reduced "short"
version of the script, which only evaluates GPT-2-small, GPT-2-medium,
Pythia-160M and Pythia-410M on ARC-c, ARC-e, BoolQ, SciQ, WinoGrande, LAMBADA
(OpenAI). The full run will take a few days to complete, whereas this reduced
setup only takes a few hours.

We run this reduced setup with

```console
python run_3.1.py --short
```

We include this reduced setup in the reproduce.sh for consistency. To
re-iterate, the script in principle is capable of running the full setup, but we
only run it for this reduced setup for the sake of time and resources.

### How it works

This script calls the `evaluate` function from the `harness.py` file. The
`evaluate` function evaluates a particular model on a set of tasks using the
Eleuther lm_eval harness, as is done in the paper.

To be able to do this, we have a wrapper `CFGLM` class, which wraps a
huggingface transformers model and adds classifier free guidance logic for the
calculation of the log-likelihood and for generation. The classifier free
guidance logic is implemented in `cfg.py`.

## Section 3.2

The `run_3.2.py` script can be used to reproduce the results of section 3.2. The
script uses the same underlying logic as the `run_3.1.py` script, but evaluates
different models (WizardLM-30B and Guanaco-65B) on different tasks using
self-consistency chain of thought prompting. Additionally, more guidance scales
are considered, to match the results of Figure 2 in the paper. The results are
saved in the folder `results/3.1/`.

We omit the running of the script from the `reproduce.sh` script (it is
commented out) because we do not have the hardware to run the two models, which
are too large to be loaded into an NVIDIA A10 GPU's memory.

### How it works

As mentioned, it works the same as the `run_3.1.py` script, but with different
models, guidance scales and tasks.
