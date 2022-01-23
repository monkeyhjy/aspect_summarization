# Source Code

## Environment

CPU：16G

GPU：RTX3080(GPU memory >= 6G)

python3.7+cuda10.2

## Setup

### pip

pip install -r requirements.txt

### conda

#### Linux

while read requirement; do conda install --yes $requirement; done < requirements.txt

#### Windows

FOR /F "delims=~" %f in (requirements.txt) DO conda install --yes "%f" || pip install "%f"

## Aspect extraction

1. run 'aspect_extraction\run_model_train.py' to classify aspects.

2. run 'aspect_extraction\run_model_test.py' to evaluate effect of KABAE, ABAE and MATE.

DKAAE in the code refers to the KABAE model. The default model in code is KABAE.

To check the result, open **aspect_extraction\out\SO\final.txt** to check Precision, Recall and F1 of model.

### Test different models

To change the model, you need to change following files:

    1. line 208 of 'aspect_extraction\run_model_test.py': config = configuration.get_config_MODEL_NAME()(The MODEL_NAME is one of following strings: DKAAE, ABAE and MATE)

    2. line 272 of 'aspect_extraction\run_model_test.py': model_name = "MODEL_NAME"(The MODEL_NAME is one of following strings: DKAAE, ABAE and MATE).

    3. line 1 of 'aspect_extraction\libs\configuration.py': current_model = "MODEL_NAME"(The MODEL_NAME is one of following strings: DKAAE, ABAE and MATE).

Tips: The evaluation of ABAE cannot be completed automatically. You can follow these steps to evaluate: 

    1. Read the topics file of ABAE output(**aspect_extraction\out\SO\ABAE\topic.txt**).

    2. Correspond the aspects of ABAE output with those marked in this paper.

    3. Write the corresponding relationship into ABAE_map array of 'aspect_extraction\evaluation.py'.

    4. Run 'aspect_extraction\evaluation.py'. 

    5. Check the weighted avg printed in console.

### Evaluate aspect attention and keyword attention weight of KABAE

This evaluation is only for KABAE.

#### aspect attention

To change the aspect attention, you need to change following files:

    1. No attention: 

        line 67 of 'aspect_extraction\libs\configuration.py': set the 'attention' as false.

        'aspect_extractionrun_model_test.py': annotate line 115, 116.

    2. Use origin attention: 

        line 67 of 'aspect_extraction\libs\configuration.py': set the 'attention' as true.

        'aspect_extractionrun_model_test.py': restore line 115, 116. (Not annotate them).

    3. Use aspect attention:

        line 67 of 'aspect_extraction\libs\configuration.py': set the 'attention' as true.

        'aspect_extractionrun_model_test.py': annotate line 115, 116.

   

#### keyword attention weight

To change the keyword attention weight, you need to change following files:

    1. Fixed weight

        line 79 of 'aspect_extraction\libs\configuration.py': set the 'aspect_encoder' as "FixedEncoder"

    2. Learnable weight

        line 79 of 'aspect_extraction\libs\configuration.py': set the 'aspect_encoder' as "WeightEncoder"

    3. attention weight

        line 79 of 'aspect_extraction\libs\configuration.py': set the 'aspect_encoder' as "WeightEncoder"

## Emotion Analysis

### Evaluate the result of Bert models

Run 'sentiment_analysis\transformer\evaluate.py' and check the accuracy in output\eval-AS.txt.

To use original bert, you need set model_path as "bert-base-uncased" and set model_name as "bert-base-uncased".

To use bert in paper, you need to set model_path as ""test-mlm" folder of your computer and set model_name as "test-mlm".

The test-mlm model can be downloaded by this url: https://drive.google.com/file/d/1EFP0XZa6WL2RJ_DhsR6GRnM4RwHNeLdE/view?usp=sharing

### Abstract summarization

1. Run 'bertsumm\run_test.py'

2. Check the result in the folder: bertsumm\data\result\bert\test

# Others

aspect_extraction:  The code of aspect extraction

bertsumm: The code of abstract summarization

dataset: part of original data(e.g. sentences of discussion and description)

output: The result of aspect extraction and emotion analysis

preprocessed_data: Include all kinds of data after preprocessing, such as sentences and keywords

sentiment_analysis: The code of emotion analysis


