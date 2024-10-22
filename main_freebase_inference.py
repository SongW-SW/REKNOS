from tqdm import tqdm
import argparse
from utils import *
from freebase_func import *
import random
from client import *
import numpy as np

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

    head_relations = list()

    relation_file = json.load(open('./relations_{}_new_dict.json'.format(args.dataset)))


    def generate_answer():

        j = 0
        for data in tqdm(datas):
            question = data[question_string]

            try:
                relations = relation_file[question]['selected_relations']
            except:
                continue

            if len(relations) > 20:
                index = np.random.choice(list(range(len(relations))), 20, replace=False)
                relations = [relations[i] for i in index]

            result = generate_answer(question, [relations], args)

            save_2_jsonl(question, result, relation_file[question]['selected_relations'],
                         file_name=args.dataset + '_result')

    
    # retrieve the entities covered by "selected_relations"
    def retrieve_entities(hop=3):
        
        

        j=0
        relation_entity_file=open('./relation_files/relation_entity_{}_{}hop_split_{}.json'.format(args.dataset, hop, j), 'w')
        relation_entity={}
        for data in tqdm(datas):
            question = data[question_string]
            try:
                relations = relation_file[question]['selected_relations']
            except:
                continue

            relation_entity[question] = {}

            topic_entity = data['topic_entity']
            
            if hop==2:
                relations = {(relation1, relation2) for relation1, relation2, _ in relations}

            if len(relations) > 100:
                index = np.random.choice(list(range(len(relations))), 100, replace=False)
                relations = [relations[i] for i in index]


            for entity in topic_entity:
                if entity != "[FINISH_ID]":
                    if hop==3:
                        for relation_1, relation_2, relation_3 in relations:
                            entities= retrieve_entity_sparql(entity,relation_1, relation_2, relation_3)
                            relation_entity[question][','.join([relation_1, relation_2, relation_3])] = {'e1':topic_entity[entity], 'es':entities}
                            #print(entities)
                            print(len(entities))
                    elif hop==2:
                        for relation_1, relation_2 in relations:
                            entities= retrieve_entity_sparql_2hop(entity,relation_1, relation_2)
                            relation_entity[question][','.join([relation_1, relation_2])] = {'e1':topic_entity[entity], 'es':entities}
                            #print(entities)
                            print(len(entities))
                    
                    #retreive all entities at same time
                    elif hop==5:
                        entities= retrieve_entity_sparql_total(entity,relations[:3])
                        relation_entity[question][','.join([relation_1, relation_2])] = {'e1':topic_entity[entity], 'es':entities}
                            #print(entities)
                        print(len(entities))
                        
                    break

            if len(relation_entity)%200==0: #50
                json.dump(relation_entity, relation_entity_file)
                j=j+1
                relation_entity_file = open('./relation_files/relation_entity_{}_{}hop_split_{}.json'.format(args.dataset, hop, j), 'w')
                relation_entity = {}

        json.dump(relation_entity, relation_entity_file)

    retrieve_entities()


#with open('./relation_entity_grailqa.json', 'r') as file:
#    data = file.read()
#    data = f"[{data.replace('}{', '},{')}]"
#    try:
#        json_data = json.loads(data)
#        i=0
#        for obj in json_data:
#            i += 1
#        print(i)
#    except json.JSONDecodeError as e:
#        print(f"Error decoding JSON: {e}")
