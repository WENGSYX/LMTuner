## Example: Medial Dataset

**We will demonstrate how to train Llama-7B on the MedDialog dataset using LMTuner.**


### Interaction
Step 1, determine the parameters needed for training through automated dialogue with GPT-4.

(We also provide `./ARGS.json` to skip this step)

```python
from LMTuner import Let_Tune
Let_Tune()
```

### Training

After obtaining `ARGS.json`, we will automatically start training. The MedDialog training set has been pre-converted to the data.jsonl file.
```python
from LMTuner import Let_Tune
Let_Tune(ARGS_file='./ARGS.json')
```

### Inference

We provide a convenient way for inference:
```bash
python -m LMTuner.inference --ARGS ARGS.json --generate_file ./test.jsonl
```

Or you can use `model.generate` for model inference.
```python
from LMTuner.models import get_model_and_tokenizer

model,tokenizers = get_model_and_tokenizer(args)
model.generate(['Hello'],tokenizer)
```