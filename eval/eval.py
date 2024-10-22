import argparse
import json
import itertools
from utils import *

def jsonl_to_json(jsonl_file, json_file):
    with open(jsonl_file, 'r') as infile:
        with open(json_file, 'w') as outfile:
            json_lines = infile.readlines()
            json_list = [json.loads(line) for line in json_lines]
            json.dump(json_list, outfile, indent=4)

def concatenate_elements(nested_list):
    result = ""
    for outer_list in nested_list:
        for inner_list in outer_list:
            for triplet in inner_list:
                print(triplet)
                result += " ".join(triplet) + " "
    return result.strip()

def split_and_flatten(input_list):
    result = []
    for item in input_list:
        # Split the string by spaces and extend the result list
        result.extend(item.split(" "))
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="cwq", help="choose the dataset.")
    parser.add_argument("--output_file", type=str,
                        default="ToG_cwq.json", help="the output file name.")
    parser.add_argument("--constraints_refuse", type=bool,
                        default=True, help="LLM may have refuse erorr, enable this option to skip current sample.")
    args = parser.parse_args()
    
    jsonl_to_json(args.output_file, args.output_file.replace('.jsonl','.json'))

    ground_truth_datas, question_string = prepare_dataset_for_eval(args.dataset)

    num_right = 0
    num_error = 0
    
    
    #output_datas=json.load(open(args.output_file.replace('.jsonl','.json'), encoding='utf-8'))
    
    if args.dataset=='grailqa':
        relation_set=set()
        for one in ground_truth_datas:
            for relation in one['graph_query']['edges']:
                relation_set.add('_'.join(relation['relation'].split('.')[:1]))

        #print(relation_set)
        print(len(relation_set))
        #print(1/0)
    
    relation_entity={}
    
    for i in range(100):
        try:
            relation_entity_file=json.load(open('../ToG/relation_files/relation_entity_{}_3hop_split_{}.json'.format(args.dataset, i)))
            relation_entity.update(relation_entity_file)
        except:
            break

    
 
    question_set=set()
    total_length=0
    for i,question in enumerate(relation_entity.keys()):

        
        
        if question not in question_set:
            question_set.add(question)
        else:
            continue
            
        #data['reasoning_chains']=[[data['reasoning_chains']]]
        
        #print(relation_entity[data['question']].keys())
        
        relation_num=50
        ent_num=50
        
        last_entity=[[entity_3_hop[2] for entity_3_hop in es_list['es'][:ent_num]] for key, es_list in relation_entity[question].items()]
        entity_set1 = set(itertools.chain.from_iterable(last_entity[:relation_num]))
        print('3-hop last enty set length', len(entity_set1))
        
        last_entity=[[entity_3_hop[1] for entity_3_hop in es_list['es'][:ent_num]] for key, es_list in relation_entity[question].items()]
        entity_set2 = set(itertools.chain.from_iterable(last_entity[:relation_num]))
        print('2-hop last enty set length', len(entity_set2))
        
        last_entity=[[entity_3_hop[0] for entity_3_hop in es_list['es'][:ent_num]] for key, es_list in relation_entity[question].items()]
        entity_set3 = set(itertools.chain.from_iterable(last_entity[:relation_num]))
        print('1-hop last enty set length', len(entity_set3))
        
        #entity_string=','.join(list(entity_set1))+','.join(list(entity_set2))+','.join(list(entity_set3))
        entity_string=','.join( list(set(list(entity_set1)+list(entity_set2)+list(entity_set3))))
        
        
        #entities=[[','.join(entity_3_hop) for entity_3_hop in es_list['es'][:10]] for key, es_list in relation_entity[question].items()]
        #entity_string=','.join([','.join(en_string_list) for en_string_list in entities[:100]])
        print('last enty string length', len(entity_string))
        total_length+=len(entity_string)
        
        

        
        print(len(question_set)-1)
        
        #print(data)
        answers = align(args.dataset, question_string, question, ground_truth_datas)
        #answers=split_and_flatten(answers)
        
        #if concatenate_elements(data['reasoning_chains'])=='':continue
        
        
        results =entity_string #+concatenate_elements(data['reasoning_chains']) # +data['results']
        #print('reasoning------------------\n', concatenate_elements(data['reasoning_chains']), entity_string)
        print('question------------------\n', question)
        if check_string(results):
            response=results
            #response = clean_results(results)
            if response=="NULL":
                response = results
            else:
                if exact_match(response, answers):
                    num_right+=1
                else:
                    num_error+=1
        else:
            response = results
            if args.constraints_refuse and check_string(response):
                continue
            if exact_match(response, answers):
                num_right+=1
            else:
                num_error+=1

    print("Exact Match: {}".format(float(num_right/(num_right+num_error))))
    print("right: {}, error: {}".format(num_right, num_error))
    print('avg length {}'.format(total_length/(num_right+num_error)))

    save_result2json(args.dataset, num_right, num_error, len(relation_entity_file))
    