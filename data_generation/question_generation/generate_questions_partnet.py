from __future__ import print_function
import argparse, json, os, itertools, random, shutil
import time
import re
from tqdm import tqdm
import question_engine as qeng

parser = argparse.ArgumentParser()

# Inputs
parser.add_argument('--input_scene_files', default='/home/evelyn/Desktop/partnet-reasoning/real_final_datasets/train/scenes_new',
    help="JSON file containing ground-truth scene information for all images " +
         "from render_images.py")
parser.add_argument('--metadata_file', default='metadata_partnet.json',
    help="JSON file containing metadata about functions")
parser.add_argument('--synonyms_json', default='synonyms.json',
    help="JSON file defining synonyms for parameter values")
parser.add_argument('--template_dir', default='PARTNET_templates',
    help="Directory containing JSON templates for questions")
parser.add_argument('--output_dir', default='/home/evelyn/Desktop/partnet-reasoning/real_final_datasets/train/questions',
    help="Directory containing JSON templates for questions")
# parser.add_argument('--new_scene_dir', default='../try_nscl/train/scenes_renew',
#     help="Directory containing JSON templates for questions")

# Output
parser.add_argument('--output_questions_file',
    default='/home/evelyn/Desktop/partnet-reasoning/real_final_datasets/train/PARTNET_questions.json',
    help="The output file to write containing generated questions")

# Control which and how many images to process
parser.add_argument('--scene_start_idx', default=0, type=int,
    help="The image at which to start generating questions; this allows " +
         "question generation to be split across many workers")
parser.add_argument('--num_scenes', default=0, type=int,
    help="The number of images for which to generate questions. Setting to 0 " +
         "generates questions for all scenes in the input file starting from " +
         "--scene_start_idx")

# Control the number of questions per image; we will attempt to generate
# templates_per_image * instances_per_template questions per image.
parser.add_argument('--templates_per_image', default=10, type=int,
    help="The number of different templates that should be instantiated " +
         "on each image")
parser.add_argument('--instances_per_template', default=1, type=int,
    help="The number of times each template should be instantiated on an image")

# Misc
parser.add_argument('--reset_counts_every', default=6000, type=int,
    help="How often to reset template and answer counts. Higher values will " +
         "result in flatter distributions over templates and answers, but " +
         "will result in longer runtimes.")
parser.add_argument('--verbose', action='store_true',
    help="Print more verbose output")
parser.add_argument('--time_dfs', action='store_true',
    help="Time each depth-first search; must be given with --verbose")
parser.add_argument('--profile', action='store_true',
    help="If given then run inside cProfile")
# args = parser.parse_args()


def precompute_filter_options(scene_struct, metadata, template):
  # Keys are tuples (size, color, shape, material) (where some may be None)
  # and values are lists of object idxs that match the filter criterion
  attribute_map = {}

  attr_keys = ['Object-Category', 'Part-Category', 'Part-Count', 'Color']
  common_parts = [
      "leg", "back", "central support", "pedestal", "leg bar", "wheel", "door", "body"
    ]
  categories = [obj['category'] for obj in scene_struct['objects']]
  if ('Chair' in categories and not 'Bed' in categories) or ('Bed' in categories and not 'Chair' in categories): common_parts.remove("back"); 
  if ('Chair' in categories and not 'Table' in categories) or ('Table' in categories and not 'Chair' in categories): common_parts.remove("central support"); common_parts.remove("pedestal")
  if ('Cart' in categories and not 'Refrigerator' in categories) or ('Refrigerator' in categories and not 'Cart' in categories): common_parts.remove("body")
  if ('Table' in categories and not 'Refrigerator' in categories) or ('Refrigerator' in categories and not 'Table' in categories): common_parts.remove("door")
  if ('Cart' in categories and not 'Chair' in categories) or ('Chair' in categories and not 'Cart' in categories): common_parts.remove("wheel")
  # Precompute masks
  masks = []
  for i in range(2 ** len(attr_keys)):
    mask = []
    for j in range(len(attr_keys)):
      mask.append((i // (2 ** j)) % 2)
    if not (((mask[2] == 1 or mask[3] == 1) and mask[1] == 0)): 
      masks.append(mask)

  for object_idx, obj in enumerate(scene_struct['objects']):
    part_color = obj['part_color_occluded']
    part_count = obj['part_count_occluded']
    for part in part_color.keys():
      key = [0,0,0,0]
      key[0] = obj['category']
      key[3] = {part: part_color[part][0]}
      key[2] = {part: int(part_count[part]) if part in part_count.keys() else 1}
      key[1] = part
      # keys = [tuple(obj[k] for k in attr_keys)]
      if ("perpendicular to the back wall" in template['text'][0] or "perpendicular to the left wall" in template['text'][0] or "parallel to the ground" in template['text'][0]) and part in ['seat', 'top']:
        continue
      
      for mask in masks:
        if (("many" in template['text'][0]) or ("number" in template['text'][0])) and mask[2] == 1: 
          continue
        if part == "arm horizontal bar" and mask[2] == 1:
          continue
        masked_key = []

        if (mask == [1,1,0,0] or mask == [1,1,1,0]) and part in ["seat", "body", "back", "sleep area", "top"]: continue
        if mask == [1,1,0,0] and part == "wheel": continue
        if mask == [0,0,0,0] and template['text'][0] == "How many <S>s with <CT> <CL> <P> are there?": print ("easy"); continue
        if mask[0] == 0 and mask[1] == 1 and (not part in common_parts):
          continue
        for a,b in zip(key, mask):         
          if b == 1:
            masked_key.append(a)
          else:
            masked_key.append(None)
        masked_key = tuple(masked_key)
        masked_key = json.dumps(masked_key)
        if masked_key not in attribute_map:
          attribute_map[masked_key] = set()          
        attribute_map[masked_key].add(object_idx)

  scene_struct['_filter_options'] = attribute_map


def find_filter_options(object_idxs, scene_struct, metadata, template):
  # Keys are tuples (size, color, shape, material) (where some may be None)
  # and values are lists of object idxs that match the filter criterion

  if '_filter_options' not in scene_struct:
    precompute_filter_options(scene_struct, metadata, template)

  attribute_map = {}
  object_idxs = set(object_idxs)
  for k, vs in scene_struct['_filter_options'].items():
    attribute_map[k] = sorted(list(object_idxs & vs))

  return attribute_map


def add_empty_filter_options(attribute_map, metadata, num_to_add, template):
  # Add some filtering criterion that do NOT correspond to objects

  attr_keys = ['Object-Category', 'Part-Category', 'Part-Count', 'Color']
  
  attr_vals = []
  for obj in ['Chair', 'Table', 'Refrigerator', 'Bed', 'Cart', None]:
    if obj == None: 
      parts = [
      "leg", "back", "central support", "pedestal", "leg bar", "wheel", "door", "body"
      ]
    else:
      parts = metadata['types']["Object-Part-Category"][obj]

    attr_vals.append([obj, None, None, None])

    for a in attribute_map:
      a = json.loads(a)
      if len(a) and a[0] == obj:
        part = random.choice(parts)
        attr_vals.append([obj, part, None, {part: random.choice(metadata['types']['Color'])}])
  
  for a in attribute_map:
      a = json.loads(a)
      if len(a) and a[1] != None:
        attr_vals.append([a[0], a[1], a[2], {a[1]: random.choice(metadata['types']['Color'])}])

  target_size = len(attribute_map) + num_to_add
  while len(attribute_map) < target_size:
    k = random.choice(attr_vals)
    k = tuple(k)
    k = json.dumps(k)
    if k not in attribute_map:
      attribute_map[k] = []

def find_relate_filter_options(object_idx, scene_struct, metadata, template,
    unique=False, include_zero=False, trivial_frac=0.1):
  options = {}
  if '_filter_options' not in scene_struct:
    precompute_filter_options(scene_struct, metadata)

  trivial_options = {}
  for relationship in scene_struct['relationships']:
    if relationship in ['above', 'below'] and (not "stable" in template["text"][0]): continue
    related = set(scene_struct['relationships'][relationship][object_idx])
    for filters, filtered in scene_struct['_filter_options'].items():
      intersection = related & filtered
      trivial = (intersection == filtered)
      if unique and len(intersection) != 1: continue
      if not include_zero and len(intersection) == 0: continue
      if trivial:
        trivial_options[(relationship, filters)] = sorted(list(intersection))
      else:
        options[(relationship, filters)] = sorted(list(intersection))

  N, f = len(options), trivial_frac
  num_trivial = int(round(N * f / (1 - f)))
  trivial_options = list(trivial_options.items())
  random.shuffle(trivial_options)
  for k, v in trivial_options[:num_trivial]:
    options[k] = v

  return options


def node_shallow_copy(node):
  new_node = {
    'type': node['type'],
    'inputs': node['inputs'],
  }
  if 'side_inputs' in node:
    new_node['side_inputs'] = node['side_inputs']
  return new_node


def other_heuristic(text, param_vals):
  """
  Post-processing heuristic to handle the word "other"
  """
  if ' other ' not in text and ' another ' not in text:
    return text
  target_keys = {
    '<Z>',  '<C>',  '<M>',  '<S>',
    '<Z2>', '<C2>', '<M2>', '<S2>',
  }
  if param_vals.keys() != target_keys:
    return text
  key_pairs = [
    ('<Z>', '<Z2>'),
    ('<C>', '<C2>'),
    ('<M>', '<M2>'),
    ('<S>', '<S2>'),
  ]
  remove_other = False
  for k1, k2 in key_pairs:
    v1 = param_vals.get(k1, None)
    v2 = param_vals.get(k2, None)
    if v1 != '' and v2 != '' and v1 != v2:
      print('other has got to go! %s = %s but %s = %s'
            % (k1, v1, k2, v2))
      remove_other = True
      break
  if remove_other:
    if ' other ' in text:
      text = text.replace(' other ', ' ')
    if ' another ' in text:
      text = text.replace(' another ', ' a ')
  return text


def instantiate_templates_dfs(scene_struct, template, metadata, answer_counts,
                              synonyms, max_instances=None, verbose=False):
  # print (template)
  param_name_to_type = {p['name']: p['type'] for p in template['params']} 

  initial_state = {
    'nodes': [node_shallow_copy(template['nodes'][0])],
    'vals': {},
    'input_map': {0: 0},
    'next_template_node': 1,
  }
  states = [initial_state]
  final_states = []

  while states:
    state = states.pop()
    # Check to make sure the current state is valid
    q = {'nodes': state['nodes']}

    # print (q)
    outputs = qeng.answer_question(q, metadata, scene_struct, all_outputs=True)

    answer = outputs[-1]
    
    if answer == '__INVALID__': 
      continue

    # Check to make sure constraints are satisfied for the current state
    skip_state = False
    for constraint in template['constraints']:
      if constraint['type'] == 'COMMON_CAT':
        s1 = constraint['params'][0]
        s2 = constraint['params'][1]
        v1, v2 = state['vals'].get(s1), state['vals'].get(s2)
        if v1 is not None and v2 is not None and v1 != "" and v2 != "" and (v1 in ['door', 'body'] and v2 not in ['door', 'body']) or ((v1 not in ['door', 'body'] and v2 in ['door', 'body'])):
          skip_state = True
          break

        if len (constraint['params']) > 2:
          s1 = constraint['params'][2]
          s2 = constraint['params'][3]
          v3, v4 = state['vals'].get(s1), state['vals'].get(s2)
          if (v3 is not None) and (v4 is not None) and (v3 != "thing") and (v4 != "thing"):
            if not len(list( (set(metadata['types']["Object-Part-Category"][v3]) & set(metadata['types']["Object-Part-Category"][v4])))):
              print (v3, v4)
              skip_state = True
              break

          if v1 is not None and v1 != "" and v4 is not None and v4 != "thing":
            if v1 not in metadata['types']["Object-Part-Category"][v4]: 
              print (v1,v4)
              skip_state = True
              break

          if v2 is not None and v2 != "" and v3 is not None and v3 != "thing":
            if v2 not in metadata['types']["Object-Part-Category"][v3]: 
              print (v2,v3)
              skip_state = True
              break

      elif constraint['type'] == 'COMMON_CAT2':
        s1 = constraint['params'][0]
        s2 = constraint['params'][1]
        v1, v2 = state['vals'].get(s1), state['vals'].get(s2)

        if v2 is not None and v2 != "" and v1 is not None and v1 != "thing":
          if v2 not in metadata['types']["Object-Part-Category"][v1]: 
            print ("cat2", v2,v1)
            skip_state = True
            break

      elif constraint['type'] == 'NEQ':
        p1, p2 = constraint['params']
        v1, v2 = state['vals'].get(p1), state['vals'].get(p2)
        if v1 is not None and v2 is not None and v1 != v2:
          if verbose:
            print('skipping due to NEQ constraint')
            print(constraint)
            print(state['vals'])
          skip_state = True
          break
      elif constraint['type'] == 'NULL':
        p = constraint['params'][0]
        p_type = param_name_to_type[p]
        v = state['vals'].get(p)
        if v is not None:
          skip = False
          if p_type == 'Object-Category' and v != 'thing': skip = True
          if p_type != 'Object-Category' and v != '': skip = True
          if skip:
            if verbose:
              print('skipping due to NULL constraint')
              print(constraint)
              print(state['vals'])
            skip_state = True
            break
      elif constraint['type'] == 'OUT_NEQ':
        i, j = constraint['params']
        i = state['input_map'].get(i, None)
        j = state['input_map'].get(j, None)
        if i is not None and j is not None and outputs[i] == outputs[j]:
          if verbose:
            print('skipping due to OUT_NEQ constraint')
            print(outputs[i])
            print(outputs[j])
          skip_state = True
          break
      elif constraint['type'] == 'ANALOGY':
        i, j, k, m = constraint['params']
        i = state['input_map'].get(i, None)
        j = state['input_map'].get(j, None)
        k = state['input_map'].get(k, None)
        m = state['input_map'].get(m, None)
        if i is not None and j is not None and k is not None and m is not None and ((outputs[i][2] == outputs[j][2] and outputs[i][0] == outputs[j][0]) or outputs[k][2] == outputs[m]):
          if verbose:
            print('skipping due to ANALOGY constraint')
            print(outputs[i])
            print(outputs[j])
          skip_state = True
          break
        if i is not None and j is not None and k is not None and m is not None and ((outputs[i][2] == outputs[k][2] and outputs[i][0] == outputs[k][0] and outputs[j][2] == outputs[m]) or (outputs[j][2] == outputs[k][2] and outputs[j][0] == outputs[k][0] and outputs[i][2] == outputs[m])):
          if verbose:
            print('skipping due to ANALOGY constraint')
            print(outputs[i])
            print(outputs[j])
          skip_state = True
          break
      else:
        assert False, 'Unrecognized constraint type "%s"' % constraint['type']

    if skip_state:
      continue

    # We have already checked to make sure the answer is valid, so if we have
    # processed all the nodes in the template then the current state is a valid
    # question, so add it if it passes our rejection sampling tests.
    if state['next_template_node'] == len(template['nodes']):
      # Use our rejection sampling heuristics to decide whether we should
      # keep this template instantiation
      cur_answer_count = answer_counts[answer]

      # print ("done")
      answer_counts_sorted = sorted(answer_counts.values())

      median_count = answer_counts_sorted[len(answer_counts_sorted) // 2]
      median_count = max(median_count, 5)

      if cur_answer_count > 1.1 * answer_counts_sorted[-2]:
        if verbose: print('skipping due to second count')
        continue
      if cur_answer_count > 1.5 * median_count:
        if verbose: print('skipping due to median')
        continue

      # If the template contains a raw relate node then we need to check for
      # degeneracy at the end
      has_relate = any('relate' in n['type'] for n in template['nodes'])
      if has_relate:
        degen = qeng.is_degenerate(q, metadata, scene_struct, answer=answer,
                                   verbose=verbose)
        # print ("check relate")
        if degen:
          # print ("degen")
          continue

      answer_counts[answer] += 1
      state['answer'] = answer
      final_states.append(state)
      if max_instances is not None and len(final_states) == max_instances:
        # print ("break")
        break
      continue

    # Otherwise fetch the next node from the template
    # Make a shallow copy so cached _outputs don't leak ... this is very nasty
    next_node = template['nodes'][state['next_template_node']]
    next_node = node_shallow_copy(next_node)

    special_nodes = {
        'filter_object_unique', 'filter_object_count', 'filter_object_exist', 'filter',
        'relate_filter', 'relate_filter_unique', 'relate_filter_count',
        'relate_filter_exist'
    }
    # print (next_node['type'])

    # if next_node['type'] == 'filter_object_unique':
    #   input_map = {k: v for k, v in state['input_map'].items()}
    #   print (input_map)
    #   unique_inputs = input_map[next_node['inputs'][0]]
    #   print (unique_inputs)
    #   if len(unique_inputs) == 1:
    #     continue
    

    if next_node['type'] in special_nodes:

      if next_node['type'].startswith('relate_filter'):
        unique = (next_node['type'] == 'relate_filter_unique')
        include_zero = (next_node['type'] == 'relate_filter_count'
                        or next_node['type'] == 'relate_filter_exist')
        filter_options = find_relate_filter_options(answer, scene_struct, metadata, template,
                            unique=unique, include_zero=False)
      else:
        filter_options = find_filter_options(answer, scene_struct, metadata, template)
        if next_node['type'] == 'filter':
          # Remove null filter
          filter_options.pop((None, None, None, None), None)
        if next_node['type'] == 'filter_object_unique':
          # Get rid of all filter options that don't result in a single object
          filter_options = {k: v for k, v in filter_options.items()
                            if len(v) == 1}
        else:
          # Add some filter options that do NOT correspond to the scene
          if next_node['type'] == 'filter_object_exist':
            # For filter_exist we want an equal number that do and don't
            num_to_add = len(filter_options) / 3
          elif next_node['type'] == 'filter_object_count' or next_node['type'] == 'filter':
            # For filter_count add nulls equal to the number of singletons
            num_to_add = sum(1 for k, v in filter_options.items() if len(v) == 1) / 2
          add_empty_filter_options(filter_options, metadata, num_to_add, template)

      filter_option_keys = list(filter_options.keys())
      random.shuffle(filter_option_keys)

      for k in filter_option_keys:
        try:
          k = json.loads(k)
        except:
          pass
        new_nodes = []
        cur_next_vals = {k: v for k, v in state['vals'].items()}

        next_input = state['input_map'][next_node['inputs'][0]]
        filter_side_inputs = next_node['side_inputs']
        if next_node['type'].startswith('relate'):
          param_name = next_node['side_inputs'][0] # First one should be relate
          filter_side_inputs = next_node['side_inputs'][1:]
          param_type = param_name_to_type[param_name]
          assert param_type == 'Relation'
          param_val = k[0]
          k = json.loads(k[1])
          new_nodes.append({
            'type': 'relate',
            'inputs': [next_input],
            'side_inputs': [param_val],
          })
          cur_next_vals[param_name] = param_val
          next_input = len(state['nodes']) + len(new_nodes) - 1
        for param_name, param_val in zip(filter_side_inputs, k):
          if param_name not in template["text"][0]: continue
          param_type = param_name_to_type[param_name]
          filter_type = 'filter_%s' % param_type.lower()
          if param_val is not None:
            new_nodes.append({
              'type': filter_type,
              'inputs': [next_input],
              'side_inputs': [param_val],
            })
            cur_next_vals[param_name] = param_val
            next_input = len(state['nodes']) + len(new_nodes) - 1
          elif param_val is None:
            if param_type == 'Object-Category':
              param_val = 'thing'
            else:
              param_val = ''
            cur_next_vals[param_name] = param_val

        input_map = {k: v for k, v in state['input_map'].items()}
        extra_type = None
        if next_node['type'].endswith('unique'):
          extra_type = 'unique'
        if next_node['type'].endswith('count'):
          extra_type = 'count'
        if next_node['type'].endswith('exist'):
          extra_type = 'exist'
        if extra_type is not None:
          new_nodes.append({
            'type': extra_type,
            'inputs': [input_map[next_node['inputs'][0]] + len(new_nodes)],
          })
        input_map[state['next_template_node']] = len(state['nodes']) + len(new_nodes) - 1
        states.append({
          'nodes': state['nodes'] + new_nodes,
          'vals': cur_next_vals,
          'input_map': input_map,
          'next_template_node': state['next_template_node'] + 1,
        })

    elif 'side_inputs' in next_node:
      # If the next node has template parameters, expand them out
      
      q_type = next_node["type"]

      common_parts = [
      "leg", "back", "central support", "pedestal", "leg bar", "wheel", "door", "body"
      ]
      categories = [obj['category'] for obj in scene_struct['objects']]

      if ('Chair' in categories and not 'Bed' in categories) or ('Bed' in categories and not 'Chair' in categories): common_parts.remove("back"); 
      if ('Chair' in categories and not 'Table' in categories) or ('Table' in categories and not 'Chair' in categories): common_parts.remove("central support"); common_parts.remove("pedestal")
      if ('Cart' in categories and not 'Refrigerator' in categories) or ('Refrigerator' in categories and not 'Cart' in categories): common_parts.remove("body")
      if ('Table' in categories and not 'Refrigerator' in categories) or ('Refrigerator' in categories and not 'Table' in categories): common_parts.remove("door")
      if ('Cart' in categories and not 'Chair' in categories) or ('Chair' in categories and not 'Cart' in categories): common_parts.remove("wheel")


      if ('part-color' in q_type) or ('part-count' in q_type) or ('part-category' in q_type) or ('part-geometry' in q_type) or ('change' in q_type):
        
        param_name = next_node['side_inputs'][0]
        param_type = param_name_to_type[param_name]        
        
        if 'part-color' in q_type:
          param_vals = list(scene_struct['objects'][answer]['part_color_occluded'].keys())
          param_vals2 = list(scene_struct['objects'][answer]['part_color_occluded'].values())
          param_vals2 = [a[0] for a in param_vals2]          
        elif 'part-count' in q_type:
          param_vals = list(scene_struct['objects'][answer]['part_count_occluded'].keys())
          param_vals2 = list(scene_struct['objects'][answer]['part_count_occluded'].values())
        elif 'part-category' in q_type:
          param_vals = list(scene_struct['objects'][answer]['part_color_occluded'].values())
          param_vals = [a[0] for a in param_vals]
          param_vals2 = list(scene_struct['objects'][answer]['part_color_occluded'].keys())
          
        elif 'part-geometry' in q_type:
          import numpy as np
          param_vals = []
          param_vals2 = []
          if "geometry" in scene_struct['objects'][answer]["question_type"]:
            for k in scene_struct['objects'][answer]['part_color_occluded'].keys():
              v = scene_struct['objects'][answer]['part_color_occluded'][k]
              if 'line_geo' in scene_struct['objects'][answer] and k in scene_struct['objects'][answer]['line_geo'].keys():
                if np.max(np.array(scene_struct['objects'][answer]['line_geo'][k])) >= 10000 or np.min(np.array(scene_struct['objects'][answer]['line_geo'][k])) <= -10000: continue
                param_vals.append(v)
                param_vals2.append(k)
  
              if 'plane_geo' in scene_struct['objects'][answer] and k in scene_struct['objects'][answer]['plane_geo'].keys():
                if np.max(np.array(scene_struct['objects'][answer]['plane_geo'][k])) >= 10000 or np.min(np.array(scene_struct['objects'][answer]['plane_geo'][k])) <= -10000: continue
                param_vals.append(v)
                param_vals2.append(k)

        elif 'change' in q_type:
          param_vals = ["left", "right", "front", "behind"]

        if 'change' in q_type:
          random.shuffle(param_vals)
          for val in param_vals:
            input_map = {k: v for k, v in state['input_map'].items()}
            input_map[state['next_template_node']] = len(state['nodes'])
            cur_next_node = {
              'type': next_node['type'],
              'inputs': [input_map[idx] if isinstance (idx, int) else idx for idx in next_node['inputs']],
              'side_inputs': [val],
            }
            cur_next_vals = {k: v for k, v in state['vals'].items()}
            cur_next_vals[param_name] = val

            states.append({
              'nodes': state['nodes'] + [cur_next_node],
              'vals': cur_next_vals,
              'input_map': input_map,
              'next_template_node': state['next_template_node'] + 1,
            })

        # random.shuffle(param_vals)
        else:
          if len(param_vals):
            temp = list(zip(param_vals, param_vals2))
            random.shuffle(temp)
            param_vals, param_vals2 = zip(*temp)
          keep2 = False
          for val, val2 in zip(param_vals, param_vals2):

            if "thing" in list(state['vals'].values()) and ('part-color' in q_type or 'part-count' in q_type ) and val not in common_parts: continue
            if "thing" in list(state['vals'].values()) and ('part-category' in q_type) and val2 not in common_parts: continue
            if 'part-count' in q_type and val in ["arm", 'arm horizontal bar']: continue

            keep = True

            for n, v in state['vals'].items():
              if isinstance(v, dict):
                v = str(list(v.values())[0])
              if val == v or val[0] == v:
                keep = False
              try:
                if val2 == v or val2[0] == v: keep = False
              except:
                pass

            if not keep: continue

            if val in list(state['vals'].values()) or val2 in list(state['vals'].values()): continue

            keep2 = True
            input_map = {k: v for k, v in state['input_map'].items()}
            input_map[state['next_template_node']] = len(state['nodes'])
            cur_next_node = {
              'type': next_node['type'],
              'inputs': [input_map[idx] if isinstance (idx, int) else idx for idx in next_node['inputs']],
              'side_inputs': [val],
            }

            cur_next_vals = {k: v for k, v in state['vals'].items()}
            cur_next_vals[param_name] = val

            states.append({
              'nodes': state['nodes'] + [cur_next_node],
              'vals': cur_next_vals,
              'input_map': input_map,
              'next_template_node': state['next_template_node'] + 1,
            })
          if not keep2: 
            continue
            # continue
      else:
        param_name = next_node['side_inputs'][0]
        param_type = param_name_to_type[param_name]
        param_vals = metadata['types'][param_type][:]
        random.shuffle(param_vals)
        for val in param_vals:
          input_map = {k: v for k, v in state['input_map'].items()}
          input_map[state['next_template_node']] = len(state['nodes'])
          cur_next_node = {
            'type': next_node['type'],
            'inputs': [input_map[idx] if isinstance (idx, int) else idx for idx in next_node['inputs']],
            'side_inputs': [val],
          }
          cur_next_vals = {k: v for k, v in state['vals'].items()}
          cur_next_vals[param_name] = val

          states.append({
            'nodes': state['nodes'] + [cur_next_node],
            'vals': cur_next_vals,
            'input_map': input_map,
            'next_template_node': state['next_template_node'] + 1,
          })
    else:
      input_map = {k: v for k, v in state['input_map'].items()}
      input_map[state['next_template_node']] = len(state['nodes'])
      next_node = {
        'type': next_node['type'],
        'inputs': [input_map[idx] if isinstance (idx, int) else idx for idx in next_node['inputs']],
      }
      states.append({
        'nodes': state['nodes'] + [next_node],
        'vals': state['vals'],
        'input_map': input_map,
        'next_template_node': state['next_template_node'] + 1,
      })

  # Actually instantiate the template with the solutions we've found
  text_questions, structured_questions, answers = [], [], []

  category = ""
  count = dict()
  for state in final_states:
    structured_questions.append(state['nodes'])
    answers.append(state['answer'])
    text = random.choice(template['text'])

    for name, val in state['vals'].items():
      if isinstance(val, dict):
        val = str(list(val.values())[0])
      if isinstance(val, list) or isinstance(val, tuple):
        val = val[0]
      if isinstance(val, int):
        val = str(val)
      # print (name, val)

      if 'S' in name: 
        if val == "": 
          val = "thing"
          state['vals'][name] = val
        p = name.replace('S', 'P')
        try:
          if state['vals'][p] == '': text = text.replace(name + " with", name).replace(name + "s with", name + 's')
        except:
          pass

        category = val
      if 'CT' in name and val != '': count[name] = int(val)

      if val in synonyms:
        val = random.choice(synonyms[val])

      if 'P' in name:
        if not val == '':      
          ct = name.replace('P', 'CT')
          if ct in state['vals'].keys() and state['vals'][ct] != "" :
            if int(state['vals'][ct][val]) > 1: 
              val += "s"
              text = text.replace("is the %s"%name, "are the %s"%name)
          else:
            if not name + "s" in text: 
              if val in ["arm", "leg", "leg bar", "wheel", "arm vertical bar", "arm horizontal bar", "back vertical bar", "back horizontal bar", "door", "drawer", "shelf"]:
                val += "s"
                text = text.replace("is the %s"%name, "are the %s"%name)
              if val in ["arm", "arm vertical bar", "arm horizontal bar"]: text = text.replace("a %s"%val, "an %s"%val)

      text = text.replace(name, val)
      text = ' '.join(text.split())
      
    # text = replace_optionals(text)
    # text = ' '.join(text.split())
    # text = other_heuristic(text, state['vals'])
    text = text.lower()
    # text = text.replace("leg", "visible leg")
    if "<s" in text:
      if "stable" in text: text = text.replace("<s> with <ct> <cl> <p>", "unstable object")
      text = text.replace("<s1> with <ct1> <cl1> <p1>", "object").replace("<s1>s with <ct1> <cl1> <p1>", "objects").replace("<s2> with <ct2> <cl2> <p2>", "object").replace("<s3> with <ct3> <cl3> <p3>", "thing").replace("<s2>s with <ct2> <cl2> <p2>", "things").replace("<s3>s with <ct3> <cl3> <p3>", "objects").replace("<s1>", "object").replace("<s1>s", "things").replace("<s2>", "object").replace("<s2>s", "things").replace("<s>", "thing").replace("<s>s", "objects").replace("<s> with <ct> <cl> <p>", "thing").replace("<s>s with <ct> <cl> <p>", "objects")
    # text = text.replace("leg", "visible leg").replace("wheel", "visible wheel").replace("arm", "visible arm")
    text = text.replace("make the thing stable", "make the unstable object stable").replace("make the object stable", "make the unstable object stable")
    text = text.replace("shelfs", "shelves")
    # print (text)
    text_questions.append(text)

    if answers[0] in ["red", "yellow", "blue", "green", "cyan", "gray", "brown", "purple"] and answers[0] in text_questions[0] and ("perpendicular" in text_questions[0] or "parallel" in text_questions[0] or "geome" in text_questions[0]):
      print ("easy geometry")
      return [], [], []

  return text_questions, structured_questions, answers

def replace_optionals(s):
  """
  Each substring of s that is surrounded in square brackets is treated as
  optional and is removed with probability 0.5. For example the string

  "A [aa] B [bb]"

  could become any of

  "A aa B bb"
  "A  B bb"
  "A aa B "
  "A  B "

  with probability 1/4.
  """
  pat = re.compile(r'\[([^\[]*)\]')

  while True:
    match = re.search(pat, s)
    if not match:
      break
    i0 = match.start()
    i1 = match.end()
    if random.random() > 0.5:
      s = s[:i0] + match.groups()[0] + s[i1:]
    else:
      s = s[:i0] + s[i1:]
  return s


def main(args):
  with open(args.metadata_file, 'r') as f:
    metadata = json.load(f)
    dataset = metadata['dataset']
  
  functions_by_name = {}
  for f in metadata['functions']:
    functions_by_name[f['name']] = f
  metadata['_functions_by_name'] = functions_by_name

  # Load templates from disk
  # Key is (filename, file_idx)
  num_loaded_templates = 0
  templates = {}
  for fn in os.listdir(args.template_dir):
    if not fn.endswith('.json'): continue
    with open(os.path.join(args.template_dir, fn), 'r') as f:
      base = os.path.splitext(fn)[0]
      for i, template in enumerate(json.load(f)):
        num_loaded_templates += 1
        key = (fn, i)
        templates[key] = template
  print('Read %d templates from disk' % num_loaded_templates)

  def reset_counts():
    # Maps a template (filename, index) to the number of questions we have
    # so far using that template
    template_counts = {}
    # Maps a template (filename, index) to a dict mapping the answer to the
    # number of questions so far of that template type with that answer
    template_answer_counts = {}
    node_type_to_dtype = {n['name']: n['output'] for n in metadata['functions']}
    for key, template in templates.items():
      template_counts[key[:2]] = 0
      final_node_type = template['nodes'][-1]['type']
      final_dtype = node_type_to_dtype[final_node_type]
      answers = metadata['types'][final_dtype]
      if final_dtype == 'Bool':
        answers = [True, False]
      if final_dtype == 'Integer':
        answers = list(range(0, 10))
      template_answer_counts[key[:2]] = {}
      for a in answers:
        template_answer_counts[key[:2]][a] = 0
    return template_counts, template_answer_counts

  template_counts, template_answer_counts = reset_counts()

  # Read file containing input scenes
  all_scenes = []
  # with open(args.input_scene_file, 'r') as f:
  #   scene_data = json.load(f)
  #   all_scenes = scene_data['scenes']
  #   scene_info = scene_data['info']

  all_scene_files = os.listdir(args.input_scene_files)
  for scene_file in all_scene_files:
    try:
      all_scenes.append(json.load(open(os.path.join(args.input_scene_files, scene_file))))
    except:
      pass

  begin = args.scene_start_idx
  if args.num_scenes > 0:
    end = args.scene_start_idx + args.num_scenes
    all_scenes = all_scenes[begin:end]
  else:
    all_scenes = all_scenes[begin:]

  # Read synonyms file
  with open(args.synonyms_json, 'r') as f:
    synonyms = json.load(f)

  questions = []
  scene_count = 0
  for i, scene in tqdm(enumerate(all_scenes)):
    if "question" in scene and scene["question"] == False: continue
    scene_fn = scene['image_filename']
    question_fn = scene_fn.replace(".png", "_question.json")
    new_scene_fn = scene_fn.replace(".png", "_new.json")
    scene_questions = []
    if question_fn in os.listdir(args.output_dir): continue

    scene_struct = scene

    for (i, obj) in enumerate(scene_struct['objects']):
      try:
        if obj['line_geo'] == dict() and obj['plane_geo'] == dict(): scene_struct['objects'][i]['question_type'].remove("geometry")
      except:
        obj['question_type'] = ['perception']    

    print('starting image %s (%d / %d)'
          % (scene_fn, i + 1, len(all_scenes)))

    if scene_count % args.reset_counts_every == 0:
      print('resetting counts')
      template_counts, template_answer_counts = reset_counts()
    scene_count += 1

    # Order templates by the number of questions we have so far for those
    # templates. This is a simple heuristic to give a flat distribution over
    # templates.

    templates_items = list(templates.items())
    templates_items = sorted(templates_items,
                        key=lambda x: template_counts[x[0][:2]])
    num_instantiated = 0
    for (fn, idx), template in templates_items:
      if ((not "physics" in scene_struct.keys()) or scene_struct["physics"] == False) and "physics" in fn: continue
      if "physics" in scene_struct.keys() and "geometry" in fn: continue
      if args.verbose:
        print('trying template ', fn, idx)
      if args.time_dfs and args.verbose:
        tic = time.time()
      ts, qs, ans = instantiate_templates_dfs(
                      scene_struct,
                      template,
                      metadata,
                      template_answer_counts[(fn, idx)],
                      synonyms,
                      max_instances=args.instances_per_template,
                      verbose=False)
      if args.time_dfs and args.verbose:
        toc = time.time()
        print('that took ', toc - tic)
      image_index = int(os.path.splitext(scene_fn)[0].split('_')[-1])
      for t, q, a in zip(ts, qs, ans):
        questions.append({
          'split': scene_struct['split'],
          'image_filename': scene_fn,
          'image_index': image_index,
          'image': os.path.splitext(scene_fn)[0],
          'question': t,
          'program': q,
          'answer': a,
          'template_filename': fn,
          'question_family_index': idx,
          'question_index': len(questions),
        })
        scene_questions.append({
          'split': scene_struct['split'],
          'image_filename': scene_fn,
          'image_index': image_index,
          'image': os.path.splitext(scene_fn)[0],
          'question': t,
          'program': q,
          'answer': a,
          'template_filename': fn,
          'question_family_index': idx,
          'question_index': len(questions),
        })

      if len(ts) > 0 and not "<" in ts[0] and not ">" in ts[0]:
        if args.verbose:
          print('got one!')
        num_instantiated += 1

        template_counts[(fn, idx)] += 1
      elif args.verbose:
        print('did not get any =(')
      if num_instantiated >= args.templates_per_image:
        break

    with open(os.path.join(args.output_dir, question_fn), 'w') as f:
        print('Writing output to %s' % question_fn)
        json.dump({
            # 'info': scene_info,
            'questions': scene_questions,
          }, f)

  for q in questions:
    for f in q['program']:
      if 'side_inputs' in f:
        f['value_inputs'] = f['side_inputs']
        del f['side_inputs']
      else:
        f['value_inputs'] = []

  with open(args.output_questions_file, 'w') as f:
    print('Writing output to %s' % args.output_questions_file)
    json.dump({
        # 'info': scene_info,
        'questions': questions,
      }, f)


if __name__ == '__main__':
  args = parser.parse_args()
  if args.profile:
    import cProfile
    cProfile.run('main(args)')
  else:
    main(args)

