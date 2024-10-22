from SPARQLWrapper import SPARQLWrapper, JSON
from utils import *

SPARQLPATH = "http://localhost:8890/sparql"  # depend on your own internal address and port, shown in Freebase folder's readme.md

# pre-defined sparqls
sparql_head_relations_2hop = """
PREFIX ns: <http://rdf.freebase.com/ns/>

SELECT DISTINCT ?relation1 ?relation2
WHERE {
  ns:%s ?relation1 ?intermediate .
  ?intermediate ?relation2 ?x .
}
"""
sparql_head_relations_3hop = """\nPREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?relation1 ?relation2 ?relation3\nWHERE {\n  ns:%s ?relation1 ?intermediate1 .\n  ?intermediate1 ?relation2 ?intermediate2 .\n  ?intermediate2 ?relation3 ?x .\n}"""

sparql_head_relations = """\nPREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?relation\nWHERE {\n  ns:%s ?relation ?x .\n}"""
sparql_tail_relations = """\nPREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?relation\nWHERE {\n  ?x ?relation ns:%s .\n}"""
sparql_tail_entities_extract = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?tailEntity\nWHERE {\nns:%s ns:%s ?tailEntity .\n}"""
sparql_head_entities_extract = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?tailEntity\nWHERE {\n?tailEntity ns:%s ns:%s  .\n}"""
sparql_id = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?tailEntity\nWHERE {\n  {\n    ?entity ns:type.object.name ?tailEntity .\n    FILTER(?entity = ns:%s)\n  }\n  UNION\n  {\n    ?entity <http://www.w3.org/2002/07/owl#sameAs> ?tailEntity .\n    FILTER(?entity = ns:%s)\n  }\n}"""


def check_end_word(s):
    words = [" ID", " code", " number", "instance of", "website", "URL", "inception", "image", " rate", " count"]
    return any(s.endswith(word) for word in words)


def abandon_rels(relation):
    if relation == "type.object.type" or relation == "type.object.name" or relation.startswith(
            "common.") or relation.startswith("freebase.") or "sameAs" in relation:
        return True


def execurte_sparql(sparql_query):
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    return results["results"]["bindings"]


def replace_relation_prefix_2hop(relations):
    return [(relation['relation1']['value'].replace("http://rdf.freebase.com/ns/", ""),
             relation['relation2']['value'].replace("http://rdf.freebase.com/ns/", "")) for relation in relations]


def replace_relation_prefix_3hop(relations):
    return [(relation['relation1']['value'].replace("http://rdf.freebase.com/ns/", ""),
             relation['relation2']['value'].replace("http://rdf.freebase.com/ns/", ""),
             relation['relation3']['value'].replace("http://rdf.freebase.com/ns/", "")) for relation in relations]

def replace_relation_prefix_entity(entities, hop=2):
    if hop==3:
        return [(entity['entity1']['value'].replace("http://rdf.freebase.com/ns/", ""),
                 entity['entity2']['value'].replace("http://rdf.freebase.com/ns/", ""),
                 entity['entity3']['value'].replace("http://rdf.freebase.com/ns/", "")) for entity in entities]
    elif hop==2:
                return [(entity['entity1']['value'].replace("http://rdf.freebase.com/ns/", ""),
                 entity['entity2']['value'].replace("http://rdf.freebase.com/ns/", "")) for entity in entities]
    
    
def replace_relation_prefix(relations):
    return [relation['relation']['value'].replace("http://rdf.freebase.com/ns/", "") for relation in relations]


def replace_entities_prefix(entities):
    return [entity['tailEntity']['value'].replace("http://rdf.freebase.com/ns/", "") for entity in entities]


def id2entity_name_or_type(entity_id):
    sparql_query = sparql_id % (entity_id, entity_id)
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    if len(results["results"]["bindings"]) == 0:
        return "UnName_Entity"
    else:
        return results["results"]["bindings"][0]['tailEntity']['value']


from freebase_func import *
from prompt_list import *
import json
import time
import openai
import re
from prompt_list import *
from rank_bm25 import BM25Okapi
from sentence_transformers import util
from sentence_transformers import SentenceTransformer


def clean_relations(string, entity_id, head_relations):
    pattern = r"{\s*(?P<relation>[^()]+)\s+\(Score:\s+(?P<score>[0-9.]+)\)}"
    relations = []
    for match in re.finditer(pattern, string):
        relation = match.group("relation").strip()
        if ';' in relation:
            continue
        score = match.group("score")
        if not relation or not score:
            return False, "output uncompleted.."
        try:
            score = float(score)
        except ValueError:
            return False, "Invalid score"
        if relation in head_relations:
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": True,
                              'meta_relation': relation.split('.')[0]})
        else:
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": False,
                              'meta_relation': relation.split('.')[0]})
    if not relations:
        return False, "No relations found"
    return True, relations


def if_all_zero(topn_scores):
    return all(score == 0 for score in topn_scores)


def clean_relations_bm25_sent(topn_relations, topn_scores, entity_id, head_relations):
    relations = []
    if if_all_zero(topn_scores):
        topn_scores = [float(1 / len(topn_scores))] * len(topn_scores)
    i = 0
    for relation in topn_relations:
        if relation in head_relations:
            relations.append({"entity": entity_id, "relation": relation, "score": topn_scores[i], "head": True})
        else:
            relations.append({"entity": entity_id, "relation": relation, "score": topn_scores[i], "head": False})
        i += 1
    return True, relations


def construct_relation_prune_prompt(question, entity_name, total_relations, args):
    return extract_relation_prompt % (
    args.width, args.width) + question + '\nTopic Entity: ' + entity_name + '\nRelations: ' + '; '.join(
        total_relations) + "\nA: "

def extract_meta(relation):
    return relation.split('.')[0] + '.' + relation.split('.')[1]

def construct_entity_score_prompt(question, relation, entity_candidates):
    return score_entity_candidates_prompt.format(question, relation) + "; ".join(entity_candidates) + '\nScore: '


def clean_result(string, relations):
    extracted_relations=[]
    for relation in relations:
        if relation in string:
            extracted_relations.append(relation)
    return extracted_relations

extract_relation_prompt = '''
You need to select three relations from the following candidate relations,  which are the most helpful for answering the quesiton.

Question: Mesih Pasha's uncle became emperor in what year?
Topic Entity: Mesih Pasha
Candidate Relations:
1. wiki.relation.child
2. wiki.relation.country_of_citizenship
3. wiki.relation.date_of_birth
4. wiki.relation.family
5. wiki.relation.father
6. wiki.relation.languages_spoken, written_or_signed
7. wiki.relation.military_rank
8. wiki.relation.occupation
9. wiki.relation.place_of_death
10. wiki.relation.position_held
11. wiki.relation.religion_or_worldview
12. wiki.relation.sex_or_gender
13. wiki.relation.sibling
14. wiki.relation.significant_event
A: Relation 1: wiki.relation.family (Score: 0.5): This relation is highly relevant as it can provide information about the family background of Mesih Pasha, including his uncle who became emperor.
Relation 2: wiki.relation.father (Score: 0.4): Uncle is father's brother, so father might provide some information as well.
Relation 3: wiki.relation.position held (Score: 0.1): This relation is moderately relevant as it can provide information about any significant positions held by Mesih Pasha or his uncle that could be related to becoming an emperor.


Question: {}
Topic Entity: {}
'''


def construct_prompt_relation(question, entity_name, relations, args):
    return extract_relation_prompt.format(question, entity_name)+ '\n Candidate Relations: ' + ';\n'.join(
        relations) + "\nReply the relations you selected from these candidate relations as : \n Relation 1: \n Relation 2: \n Relation 3:"


def retrieve_meta_path(entity_name, relations, relations_1, relations_2, relations_3, question, args):
    LLM_output=[]

    prompt = construct_prompt_relation(question, entity_name, relations_1, args)
    result = run_llm(prompt, args.temperature_exploration, args.max_length, args.opeani_api_keys, args.LLM_type)
    LLM_output.append(result)
    selected_relations_1=clean_result(result, relations_1)



    print(result)

    filtered_relations_2=[relation_2 for (relation_1, relation_2, relation_3) in relations if extract_meta(relation_1) in selected_relations_1]
    filtered_relations_2=list(set([extract_meta(relation_2) for relation_2 in filtered_relations_2]))

    prompt = construct_prompt_relation(question, entity_name, filtered_relations_2, args)
    result = run_llm(prompt, args.temperature_exploration, args.max_length, args.opeani_api_keys, args.LLM_type)
    LLM_output.append(result)
    selected_relations_2=clean_result(result, filtered_relations_2)



    filtered_relations_3=[relation_3 for (relation_1, relation_2, relation_3) in relations if extract_meta(relation_2) in selected_relations_2 and extract_meta(relation_1) in selected_relations_1]
    filtered_relations_3=list(set([extract_meta(relation_3) for relation_3 in filtered_relations_3]))

    prompt = construct_prompt_relation(question, entity_name, filtered_relations_3, args)
    result = run_llm(prompt, args.temperature_exploration, args.max_length, args.opeani_api_keys, args.LLM_type)
    LLM_output.append(result)
    selected_relations_3=clean_result(result, filtered_relations_3)



    selected_relations=[(relation_1, relation_2, relation_3) for (relation_1, relation_2, relation_3) in relations if extract_meta(relation_1) in selected_relations_1 and extract_meta(relation_2) in selected_relations_2 and extract_meta(relation_3) in selected_relations_3]

    return selected_relations, selected_relations_1, selected_relations_2, selected_relations_3, filtered_relations_2, filtered_relations_3, LLM_output


def relation_search_prune_2hop(entity_id, entity_name, pre_relations, pre_head, question, args):
    sparql_relations_extract_head_2hop = sparql_head_relations_3hop % (entity_id)


    try:
        head_relations_2hop = execurte_sparql(sparql_relations_extract_head_2hop)
    except:
        return [], [], [], []


    head_relations_2hop = replace_relation_prefix_3hop(head_relations_2hop)

    if args.remove_unnecessary_rel:
        head_relations_2hop = [(relation1, relation2, relation3) for relation1, relation2, relation3 in
                               head_relations_2hop if
                               not abandon_rels(relation1) and not abandon_rels(relation2) and not abandon_rels(
                                   relation3)]

    head_relations_2hop_1 = list(
        set([relation1.split('.')[0] + '.' + relation1.split('.')[1] for relation1, relation2, relation3 in
             head_relations_2hop]))
    head_relations_2hop_2 = list(
        set([relation2.split('.')[0] + '.' + relation2.split('.')[1] for relation1, relation2, relation3 in
             head_relations_2hop]))
    head_relations_2hop_3 = list(
        set([relation3.split('.')[0] + '.' + relation3.split('.')[1] for relation1, relation2, relation3 in
             head_relations_2hop]))

    print('head_relations_2hop_1 Length {}'.format(len(head_relations_2hop_1)), head_relations_2hop_1)
    print('head_relations_2hop_2 Length {}'.format(len(head_relations_2hop_2)), head_relations_2hop_2)
    print('head_relations_2hop_3 Length {}'.format(len(head_relations_2hop_3)), head_relations_2hop_3)

    return head_relations_2hop, head_relations_2hop_1, head_relations_2hop_2, head_relations_2hop_3


def retrieve_entity_sparql_total(entity, relations):
    sparql_values_block = """
VALUES (?rel1 ?rel2 ?rel3) {
""" + "\n".join([f"(ns:{rel1} ns:{rel2} ns:{rel3})" for rel1, rel2, rel3 in relations]) + """
}
"""

    sparql_retrieve_entity = f'''
    PREFIX ns: <http://rdf.freebase.com/ns/>

    SELECT DISTINCT ?entity1 ?entity2 ?entity3
    WHERE {{
      ?entity1 ?rel1 ?entity2 .
      ?entity2 ?rel2 ?entity3 .
      {sparql_values_block}
    }}
    LIMIT 1000
    '''

    sparql_retrieve_entity_query = sparql_retrieve_entity


    try:
        retrieved_entities = execurte_sparql(sparql_retrieve_entity_query)
        retrieved_entities = replace_relation_prefix_entity(retrieved_entities, hop=3)
        retrieved_entities=[ [id2entity_name_or_type(entity1), id2entity_name_or_type(entity2), id2entity_name_or_type(entity3)] for entity1, entity2, entity3 in retrieved_entities]
        return retrieved_entities
    except Exception as e:
        print(f"An error occurred: {e}")
        return []



def retrieve_entity_sparql(entity,relation_1, relation_2, relation_3):
    sparql_retrieve_entity='''
PREFIX ns: <http://rdf.freebase.com/ns/>

SELECT DISTINCT ?entity1 ?entity2 ?entity3
WHERE {
  ns:%s ns:%s ?entity1 .
  ?entity1 ns:%s ?entity2 .
  ?entity2 ns:%s ?entity3 .
}
LIMIT 500
'''

    sparql_retrieve_entity_query = sparql_retrieve_entity % (entity, relation_1, relation_2, relation_3)


    try:
        retrieved_entities = execurte_sparql(sparql_retrieve_entity_query)
        retrieved_entities = replace_relation_prefix_entity(retrieved_entities, hop=3)
        retrieved_entities=[ [id2entity_name_or_type(entity1), id2entity_name_or_type(entity2), id2entity_name_or_type(entity3)] for entity1, entity2, entity3 in retrieved_entities]
        return retrieved_entities
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def retrieve_entity_sparql_2hop(entity,relation_1, relation_2):
    sparql_retrieve_entity='''
PREFIX ns: <http://rdf.freebase.com/ns/>

SELECT DISTINCT ?entity1 ?entity2
WHERE {
  ns:%s ns:%s ?entity1 .
  ?entity1 ns:%s ?entity2 .
}
LIMIT 1000
'''

    sparql_retrieve_entity_query = sparql_retrieve_entity % (entity, relation_1, relation_2)


    try:
        retrieved_entities = execurte_sparql(sparql_retrieve_entity_query)
        retrieved_entities = replace_relation_prefix_entity(retrieved_entities, hop=2)
        retrieved_entities=[ [id2entity_name_or_type(entity1), id2entity_name_or_type(entity2)] for entity1, entity2 in retrieved_entities]
        return retrieved_entities
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
    


def relation_search_prune(entity_id, entity_name, pre_relations, pre_head, question, args):
    sparql_relations_extract_head = sparql_head_relations % (entity_id)
    head_relations = execurte_sparql(sparql_relations_extract_head)
    head_relations = replace_relation_prefix(head_relations)

    sparql_relations_extract_tail = sparql_tail_relations % (entity_id)
    tail_relations = execurte_sparql(sparql_relations_extract_tail)
    tail_relations = replace_relation_prefix(tail_relations)

    sparql_relations_extract_head_2hop = sparql_head_relations_2hop % (entity_id)
    head_relations_2hop = execurte_sparql(sparql_relations_extract_head_2hop)
    head_relations_2hop = replace_relation_prefix_2hop(head_relations_2hop)

    if args.remove_unnecessary_rel:
        head_relations = [relation for relation in head_relations if not abandon_rels(relation)]
        tail_relations = [relation for relation in tail_relations if not abandon_rels(relation)]
        head_relations_2hop = [(relation1, relation2) for relation1, relation2 in head_relations_2hop if
                               not abandon_rels(relation1) and not abandon_rels(relation2)]

    if pre_head:
        tail_relations = list(set(tail_relations) - set(pre_relations))
    else:
        head_relations = list(set(head_relations) - set(pre_relations))

    head_relations_2hop_1 = list(set([relation1.split('.')[0] for relation1, relation2 in head_relations_2hop]))
    head_relations_2hop_2 = list(set([relation2.split('.')[0] for relation1, relation2 in head_relations_2hop]))

    print('head_relations_2hop_1 Length {}'.format(len(head_relations_2hop_1)), head_relations_2hop_1)
    print('head_relations_2hop_2 Length {}'.format(len(head_relations_2hop_2)), head_relations_2hop_2)

    head_relations = list(set(head_relations))
    tail_relations = list(set(tail_relations))
    total_relations = head_relations + tail_relations
    total_relations.sort()  # make sure the order in prompt is always equal

    if args.prune_tools == "llm":
        prompt = construct_relation_prune_prompt(question, entity_name, total_relations, args)

        result = run_llm(prompt, args.temperature_exploration, args.max_length, args.opeani_api_keys, args.LLM_type)
        flag, retrieve_relations_with_scores = clean_relations(result, entity_id, head_relations)

    elif args.prune_tools == "bm25":
        topn_relations, topn_scores = compute_bm25_similarity(question, total_relations, args.width)
        flag, retrieve_relations_with_scores = clean_relations_bm25_sent(topn_relations, topn_scores, entity_id,
                                                                         head_relations)
    else:
        model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-tas-b')
        topn_relations, topn_scores = retrieve_top_docs(question, total_relations, model, args.width)
        flag, retrieve_relations_with_scores = clean_relations_bm25_sent(topn_relations, topn_scores, entity_id,
                                                                         head_relations)

    if flag:
        return retrieve_relations_with_scores
    else:
        return []  # format error or too small max_length


def entity_search(entity, relation, head=True):
    if head:
        tail_entities_extract = sparql_tail_entities_extract % (entity, relation)
        entities = execurte_sparql(tail_entities_extract)
    else:
        head_entities_extract = sparql_head_entities_extract % (entity, relation)
        entities = execurte_sparql(head_entities_extract)

    entity_ids = replace_entities_prefix(entities)
    new_entity = [entity for entity in entity_ids if entity.startswith("m.")]
    return new_entity


def entity_score(question, entity_candidates_id, score, relation, args):
    # retrieve the entity name
    entity_candidates = [id2entity_name_or_type(entity_id) for entity_id in entity_candidates_id]

    # if all unknown entity, return the score to all entities
    if all_unknown_entity(entity_candidates):
        return [1 / len(entity_candidates) * score] * len(entity_candidates), entity_candidates, entity_candidates_id
    entity_candidates = del_unknown_entity(entity_candidates)

    # if only one entity, return the score to the entity
    if len(entity_candidates) == 1:
        return [score], entity_candidates, entity_candidates_id
    # if no entity, return 0
    if len(entity_candidates) == 0:
        return [0.0], entity_candidates, entity_candidates_id

    # make sure the id and entity are in the same order
    zipped_lists = sorted(zip(entity_candidates, entity_candidates_id))
    entity_candidates, entity_candidates_id = zip(*zipped_lists)
    entity_candidates = list(entity_candidates)
    entity_candidates_id = list(entity_candidates_id)

    # retrive entity scores
    if args.prune_tools == "llm":
        prompt = construct_entity_score_prompt(question, relation, entity_candidates)

        result = run_llm(prompt, args.temperature_exploration, args.max_length, args.opeani_api_keys, args.LLM_type)

        # multiply the relation score to the entity score
        return [float(x) * score for x in
                clean_scores(result, entity_candidates)], entity_candidates, entity_candidates_id

    elif args.prune_tools == "bm25":
        topn_entities, topn_scores = compute_bm25_similarity(question, entity_candidates, args.width)
    else:
        model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-tas-b')
        topn_entities, topn_scores = retrieve_top_docs(question, entity_candidates, model, args.width)
    if if_all_zero(topn_scores):
        topn_scores = [float(1 / len(topn_scores))] * len(topn_scores)
    return [float(x) * score for x in topn_scores], topn_entities, entity_candidates_id


def update_history(entity_candidates, entity, scores, entity_candidates_id, total_candidates, total_scores,
                   total_relations, total_entities_id, total_topic_entities, total_head):
    if len(entity_candidates) == 0:
        entity_candidates.append("[FINISH]")
        entity_candidates_id = ["[FINISH_ID]"]
    candidates_relation = [entity['relation']] * len(entity_candidates)
    topic_entities = [entity['entity']] * len(entity_candidates)
    head_num = [entity['head']] * len(entity_candidates)
    total_candidates.extend(entity_candidates)
    total_scores.extend(scores)
    total_relations.extend(candidates_relation)
    total_entities_id.extend(entity_candidates_id)
    total_topic_entities.extend(topic_entities)
    total_head.extend(head_num)
    return total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head


def half_stop(question, cluster_chain_of_entities, depth, args):
    print("No new knowledge added during search depth %d, stop searching." % depth)
    answer = generate_answer(question, cluster_chain_of_entities, args)
    save_2_jsonl(question, answer, cluster_chain_of_entities, file_name=args.dataset)


def generate_answer(question, cluster_chain_of_entities, args):
    prompt = answer_prompt + question + '\n'
    chain_prompt = '\n'.join(
        [', '.join([str(x) for x in chain]) for sublist in cluster_chain_of_entities for chain in sublist])
    prompt += "\nKnowledge Triplets: " + chain_prompt + 'A: '
    result = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type)
    return result


def entity_prune(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores,
                 args):
    # sort the entities by scores

    zipped = list(
        zip(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores))
    sorted_zipped = sorted(zipped, key=lambda x: x[5], reverse=True)
    sorted_entities_id, sorted_relations, sorted_candidates, sorted_topic_entities, sorted_head, sorted_scores = [x[0]
                                                                                                                  for x
                                                                                                                  in
                                                                                                                  sorted_zipped], [
        x[1] for x in sorted_zipped], [x[2] for x in sorted_zipped], [x[3] for x in sorted_zipped], [x[4] for x in
                                                                                                     sorted_zipped], [
        x[5] for x in sorted_zipped]

    #
    #
    entities_id, relations, candidates, topics, heads, scores = sorted_entities_id[:args.width], sorted_relations[
                                                                                                 :args.width], sorted_candidates[
                                                                                                               :args.width], sorted_topic_entities[
                                                                                                                             :args.width], sorted_head[
                                                                                                                                           :args.width], sorted_scores[
                                                                                                                                                         :args.width]
    merged_list = list(zip(entities_id, relations, candidates, topics, heads, scores))
    filtered_list = [(id, rel, ent, top, hea, score) for id, rel, ent, top, hea, score in merged_list if score != 0]
    if len(filtered_list) == 0:
        return False, [], [], [], []
    entities_id, relations, candidates, tops, heads, scores = map(list, zip(*filtered_list))

    tops = [id2entity_name_or_type(entity_id) for entity_id in tops]
    cluster_chain_of_entities = [[(tops[i], relations[i], candidates[i]) for i in range(len(candidates))]]
    return True, cluster_chain_of_entities, entities_id, relations, heads


def reasoning(question, cluster_chain_of_entities, args):
    prompt = prompt_evaluate + question
    chain_prompt = '\n'.join(
        [', '.join([str(x) for x in chain]) for sublist in cluster_chain_of_entities for chain in sublist])
    prompt += "\nKnowledge Triplets: " + chain_prompt + 'A: '

    response = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type)

    # extract the answer of yes or no
    result = extract_answer(response)
    if if_true(result):
        return True, response
    else:
        return False, response




