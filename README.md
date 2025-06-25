# REKNOS


_**Reasoning of Large Language Models over Knowledge Graphs with Super-Relations**_

Here is the code of REKNOS, published in ICLR 2025. 



## Code Instruction:

### Get started

To run experiments, first install either **Freebase** or **Wikidata** locally. The installation instructions and configuration are in the `README.md` file located within the respective folder. These instructions are obtained from [ToG](https://github.com/GasolSun36/ToG).

The required libraries are provided in `requirements.txt`.

### Running

For example, to run experiments with Freebase on dataset GrailQA, run the following command:

```sh
python ./main_freebase.py --dataset grailqa --max_length 256 --temperature_exploration 0.4 --temperature_reasoning 0 --width 3 --depth 3 --remove_unnecessary_rel True --LLM_type gpt-4o-mini --opeani_api_keys your_api_key --num_retain_entity 5 --prune_tools llm
```

