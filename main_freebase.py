from tqdm import tqdm
import argparse
from utils import *
from freebase_func import *
import random
from client import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="webqsp", help="choose the dataset.")
    parser.add_argument("--max_length", type=int,
                        default=256, help="the max length of LLMs output.")
    parser.add_argument("--temperature_exploration", type=float,
                        default=0.4, help="the temperature in exploration stage.")
    parser.add_argument("--temperature_reasoning", type=float,
                        default=0, help="the temperature in reasoning stage.")
    parser.add_argument("--width", type=int,
                        default=3, help="choose the search width of ToG.")
    parser.add_argument("--depth", type=int,
                        default=3, help="choose the search depth of ToG.")
    parser.add_argument("--remove_unnecessary_rel", type=bool,
                        default=True, help="whether removing unnecessary relations.")
    parser.add_argument("--LLM_type", type=str,
                        default="gpt-3.5-turbo", help="base LLM model.")
    parser.add_argument("--opeani_api_keys", type=str,
                        default="",
                        help="if the LLM_type is gpt-3.5-turbo or gpt-4, you need add your own openai api keys.")
    parser.add_argument("--num_retain_entity", type=int,
                        default=5, help="Number of entities retained during entities search.")
    parser.add_argument("--prune_tools", type=str,
                        default="llm", help="prune tools for ToG, can be llm (same as LLM_type), bm25 or sentencebert.")
    args = parser.parse_args()

    datas, question_string = prepare_dataset(args.dataset)
    print("Start Running ToG on %s dataset." % args.dataset)

    try:
        head_relations=json.load(open('./relations_{}_new_dict.json'.format(args.dataset)))
        
    except:
        print('no existing file')
        head_relations = dict()

    for data in tqdm(datas):
        question = data[question_string]
        
        if question in head_relations:continue
        
        topic_entity = data['topic_entity']
        cluster_chain_of_entities = []
        if len(topic_entity) == 0:
            results = generate_without_explored_paths(question, args)
            save_2_jsonl(question, results, [], file_name=args.dataset)
            continue
        pre_relations = []
        pre_heads = [-1] * len(topic_entity)

        if len(pre_heads) == 0:
            continue

        for entity in topic_entity:
            # print(pre_heads)
            # print(i)
            if len(pre_heads) == 0:
                pre_heads = [-1]
            # print('topic_entity',topic_entity)
            # print('pre heads', pre_heads, i)
            if entity != "[FINISH_ID]":
                # given an entity, retrieve relations with scores
                # retrieve_relations_with_scores = relation_search_prune(entity, topic_entity[entity], pre_relations, pre_heads[i] if i<len(pre_heads) else -1, question, args)  # best entity triplet, entitiy_id
                head_relations_2hop, head_relations_2hop_1, head_relations_2hop_2, head_relations_2hop_3 = relation_search_prune_2hop(
                    entity, topic_entity[entity], pre_relations, pre_heads[0], question, args)  # best entity triplet, entitiy_id

                if len(head_relations_2hop) == 0:
                    #head_relations.append({})
                    break

                selected_relations, selected_relations_1, selected_relations_2, selected_relations_3, filtered_relations_2, filtered_relations_3, LLM_output = retrieve_meta_path(
                    topic_entity[entity], head_relations_2hop, head_relations_2hop_1, head_relations_2hop_2,
                    head_relations_2hop_3, question, args)

                head_relations[question]={'question': question, 'entity': topic_entity[entity],
                    'relations': head_relations_2hop, 'relations_set_1hop': list(
                    set([relation1 for (relation1, relation2, relation3) in head_relations_2hop])),
                                       'relations_set_2hop': list(
                                           set([relation2 for (relation1, relation2, relation3) in
                                                head_relations_2hop])),
                                       'relations_set_3hop': list(
                                           set([relation3 for (relation1, relation2, relation3) in
                                                head_relations_2hop])),
                                       'relations_1hop': head_relations_2hop_1, 'relations_2hop': head_relations_2hop_2,
                                       'relations_3hop': head_relations_2hop_3,
                                       'selected_relations': selected_relations,
                                       'selected_relations_1': selected_relations_1,
                                       'selected_relations_2': selected_relations_2,
                                       'selected_relations_3': selected_relations_3,
                                        'filtered_relations_2': filtered_relations_2,
                                        'filtered_relations_3': filtered_relations_3,
                                       'LLM_output': LLM_output}

                break

            i += 1
        
        if len(head_relations)%100==0:
            json.dump(head_relations, open('./relations_{}_new_dict.json'.format(args.dataset), 'w'))

        continue
    json.dump(head_relations, open('./relations_{}_new_dict.json'.format(args.dataset), 'w'))

