# Eval

We use **Exact Match** as our evaluation metric.

After getting the final result file, use the following command to evaluate the results:

```sh
python eval.py \  # if you wanna use Wikidata as KG source, run main_wiki.py
--dataset cwq \ # dataset your wanna test, see ToG/data/README.md
--output_file ToG_cwq.json \ 
--constraints_refuse True
```
python eval.py --dataset grailqa --output_file ../ToG/grailqa_result.jsonl --constraints_refuse True

python eval.py --dataset grailqa --output_file ../ToG/ToG_grailqa.jsonl --constraints_refuse True


python eval.py --dataset simpleqa --output_file ../ToG/ToG_simpleqa.jsonl --constraints_refuse True

python eval.py --dataset webqsp --output_file ../ToG/ToG_webqsp.jsonl --constraints_refuse True

python eval.py --dataset webquestions --output_file ../ToG/ToG_webquestions.jsonl --constraints_refuse True


After that, you will get a result json file that contains:

```sh
{
    'dataset': 
    'method': 
    'Exact Match': 
    'Right Samples': 
    'Error Sampels': 
}
```