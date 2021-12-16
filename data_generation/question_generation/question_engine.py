import json, os, math
from collections import defaultdict

"""
Utilities for working with function program representations of questions.

Some of the metadata about what question node types are available etc are stored
in a JSON metadata file.
"""


# Handlers for answering questions. Each handler receives the scene structure
# that was output from Blender, the node, and a list of values that were output
# from each of the node's inputs; the handler should return the computed output
# value from this node.


def scene_handler(scene_struct, inputs, side_inputs):
  # Just return all objects in the scene
  return list(range(len(scene_struct['objects'])))


def make_filter_handler(attribute):
  def filter_handler(scene_struct, inputs, side_inputs):
    assert len(inputs) == 1
    assert len(side_inputs) == 1
    value = side_inputs[0]
    output = []
    
    for idx in inputs[0]:
      if attribute == 'Object-Category':
        atr = scene_struct['objects'][idx]['category']
        if value == atr or value in atr:
          output.append(idx)
      
      if attribute == 'Color':
        atr = scene_struct['objects'][idx]['part_color_occluded']
        k = list(value.keys())[0]
        v = list(value.values())[0]
        if k in atr.keys() and v == atr[k][0]:
          output.append(idx)

      if attribute == 'Part-Count':
        atr = scene_struct['objects'][idx]['part_color_occluded']

        atr2 = scene_struct['objects'][idx]['part_count_occluded']
        
        k = list(value.keys())[0]
        v = list(value.values())[0]

        if k in atr.keys():
          if k in atr2.keys():
            ct = int(atr2[k])
          else:
            ct = 1
          if v == ct:
            output.append(idx)

      if attribute == 'Part-Category':
        atr = scene_struct['objects'][idx]['part_color_occluded']
        if value in list(atr.keys()):
          output.append(idx)

    if attribute == 'Object-Category' and value != "thing" and value != "object" and inputs[0] == output:
      return '__INVALID__'
    return output
  return filter_handler

def unique_handler(scene_struct, inputs, side_inputs):
  # assert len(inputs[0]) == 1
  if len(inputs[0]) != 1:
    return '__INVALID__'
  return inputs[0][0]

def vg_relate_handler(scene_struct, inputs, side_inputs):
  assert len(inputs) == 1
  assert len(side_inputs) == 1
  output = set()
  for rel in scene_struct['relationships']:
    if rel['predicate'] == side_inputs[0] and rel['subject_idx'] == inputs[0]:
      output.add(rel['object_idx'])
  return sorted(list(output))



def relate_handler(scene_struct, inputs, side_inputs):
  assert len(inputs) == 1
  assert len(side_inputs) == 1
  relation = side_inputs[0]
  return scene_struct['relationships'][relation][inputs[0]]
    

def union_handler(scene_struct, inputs, side_inputs):
  assert len(inputs) == 2
  assert len(side_inputs) == 0
  return sorted(list(set(inputs[0]) | set(inputs[1])))


def intersect_handler(scene_struct, inputs, side_inputs):
  assert len(inputs) == 2
  assert len(side_inputs) == 0
  return sorted(list(set(inputs[0]) & set(inputs[1])))


def count_handler(scene_struct, inputs, side_inputs):
  assert len(inputs) == 1
  if len(inputs[0]) >= 10: return '__INVALID__'
  return len(inputs[0])


def make_same_attr_handler(attribute):
  def same_attr_handler(scene_struct, inputs, side_inputs):
    cache_key = '_same_%s' % attribute
    
    if cache_key not in scene_struct:
      cache = {}
      for i, obj1 in enumerate(scene_struct['objects']):
        same = []
        for j, obj2 in enumerate(scene_struct['objects']):
          if i != j and obj1["category"] == obj2["category"]:
            same.append(j)
        cache[i] = same
      scene_struct[cache_key] = cache

    cache = scene_struct[cache_key]
    assert len(inputs) == 1
    assert len(side_inputs) == 0
    return cache[inputs[0]]
  return same_attr_handler

def make_same_part_attr_handler(attribute):
  def same_attr_handler(scene_struct, inputs, side_inputs):
    part = side_inputs[0]

    i = inputs[0]
    obj1 = scene_struct['objects'][i]
    
    same = []
    for j, obj2 in enumerate(scene_struct['objects']):
      
      if i != j and part in obj2[attribute].keys() and obj1[attribute][part] == obj2[attribute][part]:
        same.append(j)

    assert len(inputs) == 1
    return same
  return same_attr_handler


def make_query_handler(attribute):
  def query_handler(scene_struct, inputs, side_inputs):
    assert len(inputs) == 1
    idx = inputs[0]

    obj = scene_struct['objects'][idx]
    
    val = obj['category']

    if type(val) == list and len(val) != 1:
      return '__INVALID__'
    elif type(val) == list and len(val) == 1:
      return val[0]
    else:
      return val
  return query_handler

def make_part_query_handler(attribute):
  def query_handler(scene_struct, inputs, side_inputs):
    assert len(inputs) == 1
    idx = inputs[0]

    if side_inputs[0] == None: return '__INVALID__'

    obj = scene_struct['objects'][idx]
    if attribute == "Color":
      val = obj['part_color_occluded'][side_inputs[0]][0]
    elif attribute == "Part-Count":
      val = obj['part_count_occluded'][side_inputs[0]]
      if val >= 10: return '__INVALID__'
    elif attribute == "Part-Category":
      vals = []
      # if side_inputs[0] in list(obj['part_color_occluded'].values()):
      for k, v in obj['part_color_occluded'].items():
        if v[0] == side_inputs[0]:
          vals.append(k)

      all_colors = []
      for k, v in obj['part_color_all'].items():
        if v[0] == side_inputs[0]:
          all_colors.append(k)      

      if len(vals) > 1 or len(all_colors) > 1:
        return '__INVALID__'
      else:
        return vals[0]
    elif attribute == "Part-Geometry":
      vals = []
      part_color = dict()
      # if side_inputs[0] in list(obj['part_color_occluded'].values()):
      for k, val in obj['part_color_occluded'].items():
        if val == side_inputs[0]:
          vals.append(k)

      all_colors = []
      for k, v in obj['part_color_all'].items():
        if v[0] == side_inputs[0]:
          all_colors.append(k)    

      if len(vals) > 1 or len(all_colors) > 1:
        return '__INVALID__'

      else:
        if vals[0] in obj['line_geo'].keys():
          return ["line", obj['line_geo'][vals[0]], idx]
        if vals[0] in obj['plane_geo'].keys():
          return ["plane", obj['plane_geo'][vals[0]], idx]
      
    return val
  return query_handler

import numpy as np

def make_physics_filter_handler(attribute):
  def filter_handler(scene_struct, inputs, side_inputs):
    output = []
    if type(inputs[0]) == list: inputs = inputs[0]
    for idx in inputs:
      if "unstability" in attribute: 
        if scene_struct['objects'][idx]["stability"] == "no": output.append(idx)
      else:
        if scene_struct['objects'][idx]["stability"] == "yes": output.append(idx)
    return output
  return filter_handler

def make_physics_query_handler(attribute):
  def query_handler(scene_struct, inputs, side_inputs):
    output = []
    if not len(inputs): return '__INVALID__'
    idx = inputs[0]

    if "unstability" in attribute: 
      if scene_struct['objects'][idx]["stability"] == "no": return True
      else: return False
    elif "stability" in attribute:
      if scene_struct['objects'][idx]["stability"] == "yes": return True
      else: return False
    elif "change" in attribute:
      if "possible_change" in scene_struct['objects'][idx].keys():
        if side_inputs[0] in scene_struct['objects'][idx]["possible_change"][0]: return True
        else: return False
      else:
        return False
    elif "direction" in attribute:
      if "possible_change" in scene_struct['objects'][idx].keys():
        
        if len(scene_struct['objects'][idx]["possible_change"][0]) != 1: return '__INVALID__'
        else:
          return scene_struct['objects'][idx]["possible_change"][0][0]
      else:
        return '__INVALID__'

  return query_handler

def query_geometric_relation_handler(scene_struct, inputs, side_inputs):
  assert len(inputs) == 2
  if inputs[0] in ["ground", "left wall", "back wall"]:
    if inputs[0] == "ground": t1 = "plane"; geo1 = [0,0,1.0]; idx1 = 10000
    if inputs[0] == "left wall": t1 = "plane"; geo1 = [1.0,0,0]; idx1 = 10000
    if inputs[0] == "back wall": t1 = "plane"; geo1 = [0,1.0,0]; idx1 = 10000
  else:
    t1, geo1, idx1 = inputs[0]
    if len(geo1) == 1: geo1 = geo1[0]
  t2, geo2, idx2 = inputs[1]

  
  if len(geo2) == 1: geo2 = geo2[0]
  # if "geometry" not in scene_struct['objects'][idx1]['question_type'] or "geometry" not in scene_struct['objects'][idx2]['question_type']: return '__INVALID__'
  g_type = ''
  if t1 == t2: 
    if abs(np.array(geo1).dot(np.array(geo2))) < 0.2 and not np.max(np.array(geo2)) and not np.max(np.array(geo1)) >= 10000.0: g_type = "perpendicular"

    if (abs(np.linalg.norm(np.array(geo1) - np.array(geo2))) < 0.2 or abs(np.linalg.norm(np.array(geo2) - np.array(geo1))) < 0.2) and (not np.max(np.array(geo2)) >= 10000.0) and not np.max(np.array(geo1)) >= 10000.0: g_type = "parallel"

  else:
    if abs(np.array(geo1).dot(np.array(geo2))) < 0.2 and not np.max(np.array(geo2)) and not np.max(np.array(geo1)) >= 10000.0: g_type = "parallel"

    if (abs(np.linalg.norm(np.array(geo1) - np.array(geo2))) < 0.2 or abs(np.linalg.norm(np.array(geo2) - np.array(geo1))) < 0.2) and (not np.max(np.array(geo2)) >= 10000.0) and not np.max(np.array(geo1)) >= 10000.0: g_type = "perpendicular"

  if g_type == '': return '__INVALID__'
  else:
    return [t1, t2, g_type]  


def query_geometric_analogy_handler(attribute):
  def query_handler(scene_struct, inputs, side_inputs):
    assert len(inputs) == 3
    t1,t2,g_type = inputs[0]
    if inputs[1] in ["ground", "left wall", "back wall"]: 
      if inputs[1] == "ground": t3 = "plane"; geo3 = [0,0,1.0]; idx3 = 10000
      if inputs[1] == "left wall": t3 = "plane"; geo3 = [1.0,0,0]; idx3 = 10000
      if inputs[1] == "back wall": t3 = "plane"; geo3 = [0,1.0,0]; idx3 = 10000
    else:
      t3, geo3, idx3 = inputs[1]
      if len(geo3) == 1: geo3 = geo3[0]
    idx4 = inputs[2]
    obj = scene_struct['objects'][idx4]
    if t3 != t1 or "geometry" not in scene_struct['objects'][idx4]['question_type']: return '__INVALID__'

    part_geos = []
    count = 0
    if (t2 == t1):
      if t2 == 'line':
        geo = obj['line_geo']
      else:
        geo = obj['plane_geo']
      for part, geo4 in geo.items():
        # print (geo4)
        if len(geo4) == 1: geo4 = geo4[0]
        if "perpendicular" in g_type:
          if abs(np.array(geo3).dot(np.array(geo4))) < 0.2 and (not np.max(np.array(geo4)) >= 10000.0) and (not np.min(np.array(geo4)) <= -10000.0):
            if part in obj['part_color_occluded'].keys():
              part_geos.append(part)
            if part in obj['part_count_occluded'].keys():
                count += obj['part_count_occluded'][part]
            else:
              if part in obj['part_color_occluded'].keys():
                count += 1
        else:
          if (abs(np.linalg.norm(np.array(geo4) - np.array(geo3))) < 0.2 or abs(np.linalg.norm(np.array(geo3) - np.array(geo4))) < 0.2) and (not np.max(np.array(geo4)) >= 10000.0) and (not np.min(np.array(geo4)) <= -10000.0):
            if part in obj['part_color_occluded'].keys():
              part_geos.append(part)
            if part in obj['part_count_occluded'].keys():
                count += obj['part_count_occluded'][part]
            else:
              if part in obj['part_color_occluded'].keys():
                count += 1

    if (t2 != t1):
      if t2 == 'line':
        geo = obj['line_geo']
      else:
        geo = obj['plane_geo']
      for part, geo4 in geo.items():
        # print (geo4)
        if len(geo4) == 1: geo4 = geo4[0]
        if "parallel" in g_type:
          if abs(np.array(geo3).dot(np.array(geo4))) < 0.2 and (not np.max(np.array(geo4)) >= 10000.0) and (not np.min(np.array(geo4)) <= -10000.0):
            if part in obj['part_color_occluded'].keys():
              part_geos.append(part)
            if part in obj['part_count_occluded'].keys():
                count += obj['part_count_occluded'][part]
            else:
              if part in obj['part_color_occluded'].keys():
                count += 1
        else:
          if (abs(np.linalg.norm(np.array(geo4) - np.array(geo3))) < 0.2 or abs(np.linalg.norm(np.array(geo3) - np.array(geo4))) < 0.2) and ((not np.max(np.array(geo4)) >= 10000.0) and (not np.min(np.array(geo4)) <= -10000.0)):
            if part in obj['part_color_occluded'].keys():
              part_geos.append(part)
            if part in obj['part_count_occluded'].keys():
                count += obj['part_count_occluded'][part]
            else:
              if part in obj['part_color_occluded'].keys():
                count += 1

    if 'count' in attribute: 
      if count >= 10: return '__INVALID__'
      else: return count
    else:
      if len(part_geos) != 1 or (not part_geos[0] in obj['part_color_occluded'].keys()): return '__INVALID__'
      else:
        return obj['part_color_occluded'][part_geos[0]][0]
  return query_handler

def query_position_handler(scene_struct, inputs, side_inputs):
  assert len(inputs) == 2
  idx1 = inputs[0]
  idx2 = inputs[1]
  rels = []
  for rel, value in scene_struct["relationships"].items():
    idx1_rel = value[idx1]
    if idx2 in idx1_rel:
      rels.append(rel)
  
  return rels

def query_position_analogy_handler(attribute):
  def query_handler(scene_struct, inputs, side_inputs):
    assert len(inputs) == 2
    rels = inputs[0]

    idx1 = inputs[1]

    same_rels = []
    for idx2 in range(len(scene_struct["objects"])):
      if idx1 == idx2: continue
      same_related = True
      for rel in rels:
        if not idx2 in scene_struct["relationships"][rel][idx1]: same_related = False
      if same_related:
        same_rels.append(idx2)

    if 'count' in attribute: return len(same_rels)
    elif 'exist' in attribute: return len(same_rels) > 0
    else:
      if len(same_rels) != 1: return '__INVALID__'
      else:
        return scene_struct["objects"][same_rels[0]]['category']
  return query_handler

def make_filter_geometry_handler(attribute):
  def filter_handler(scene_struct, inputs, side_inputs):
    assert len(inputs) == 2
    
    idx = inputs[0]
    obj = scene_struct['objects'][idx]
    if obj['category'] == 'Cart' or ('line' in attribute and obj['category'] == 'Refrigerator'): return '__INVALID__'
    if inputs[1] in ["ground", "left wall", "back wall"]:
      if inputs[1] == "ground": t = "plane"; geo1 = [0,0,1.0]; idx1 = 10000
      if inputs[1] == "left wall": t = "plane"; geo1 = [1.0,0,0]; idx1 = 10000
      if inputs[1] == "back wall": t = "plane"; geo1 = [0,1.0,0]; idx1 = 10000

      if "geometry" not in scene_struct['objects'][idx]['question_type']: return '__INVALID__'

    else:
      t, geo1, idx1 = inputs[1]
      if len(geo1) == 1: geo1 = geo1[0]
    
      if idx1 == idx or "geometry" not in scene_struct['objects'][idx]['question_type'] or "geometry" not in scene_struct['objects'][idx1]['question_type']:
        return '__INVALID__'
    count = 0
    
    part_geos = []
    if "line" in t:
      if "line" in attribute:
        for part, geo2 in obj['line_geo'].items():
          if len(geo2) == 1: geo2 = geo2[0]
          if "perpendicular" in attribute:
            if abs(np.array(geo1).dot(np.array(geo2))) < 0.2 and (not np.max(np.array(geo2)) >= 10000.0) and (not np.min(np.array(geo2)) <= -10000.0):
              part_geos.append(part)
              if part in obj['part_count_occluded'].keys():
                count += obj['part_count_occluded'][part]
              else:
                if part in obj['part_color_occluded'].keys():
                  count += 1
          else:
            if (abs(np.linalg.norm(np.array(geo1) - np.array(geo2))) < 0.2 or abs(np.linalg.norm(np.array(geo2) - np.array(geo1))) < 0.2) and (not np.max(np.array(geo2)) >= 10000.0) and (not np.min(np.array(geo2)) <= -10000.0):
              part_geos.append(part)
              if part in obj['part_count_occluded'].keys():
                count += obj['part_count_occluded'][part]
              else:
                if part in obj['part_color_occluded'].keys():
                  count += 1
      else:
        for part, geo2 in obj['plane_geo'].items():
          if len(geo2) == 1: geo2 = geo2[0]
          if "parallel" in attribute:
            if abs(np.array(geo1).dot(np.array(geo2))) < 0.2 and (not np.max(np.array(geo2)) >= 10000.0) and (not np.min(np.array(geo2)) <= -10000.0):
              part_geos.append(part)
              if part in obj['part_count_occluded'].keys():
                count += obj['part_count_occluded'][part]
              else:
                if part in obj['part_color_occluded'].keys():
                  count += 1
          else:
            if (abs(np.linalg.norm(np.array(geo1) - np.array(geo2))) < 0.2 or abs(np.linalg.norm(np.array(geo2) - np.array(geo1))) < 0.2) and (not np.max(np.array(geo2)) >= 10000.0) and (not np.min(np.array(geo2)) <= -10000.0):
              part_geos.append(part)
              if part in obj['part_count_occluded'].keys():
                count += obj['part_count_occluded'][part]
              else:
                if part in obj['part_color_occluded'].keys():
                  count += 1
      

    if "plane" in t:
      
      if "line" in attribute:
        for part, geo2 in obj['line_geo'].items():
          if len(geo2) == 1: geo2 = geo2[0]
          if "perpendicular" in attribute:
            if (abs(np.linalg.norm(np.array(geo1) - np.array(geo2))) < 0.2 or abs(np.linalg.norm(np.array(geo2) - np.array(geo1))) < 0.2) and (not np.max(np.array(geo2)) >= 10000.0) and (not np.min(np.array(geo2)) <= -10000.0):
              part_geos.append(part)
              if part in obj['part_count_occluded'].keys():
                count += obj['part_count_occluded'][part]
              else:
                if part in obj['part_color_occluded'].keys():
                  count += 1
          else:
            if abs(np.array(geo1).dot(np.array(geo2))) < 0.2 and (not np.max(np.array(geo2)) >= 10000.0) and (not np.min(np.array(geo2)) <= -10000.0):
              part_geos.append(part)
              if part in obj['part_count_occluded'].keys():
                count += obj['part_count_occluded'][part]
              else:
                if part in obj['part_color_occluded'].keys():
                  count += 1
      else:
        
        for part, geo2 in obj['plane_geo'].items():
          if len(geo2) == 1: geo2 = geo2[0]
          if "parallel" in attribute:
            if (abs(np.linalg.norm(np.array(geo1) - np.array(geo2))) < 0.2 or abs(np.linalg.norm(np.array(geo2) - np.array(geo1))) < 0.2) and (not np.max(np.array(geo2)) >= 10000.0) and (not np.min(np.array(geo2)) <= -10000.0):
              part_geos.append(part)
              if part in obj['part_count_occluded'].keys():
                count += obj['part_count_occluded'][part]
              else:
                if part in obj['part_color_occluded'].keys():
                  count += 1
          else:         
            if abs(np.array(geo1).dot(np.array(geo2))) < 0.2 and (not np.max(np.array(geo2)) >= 10000.0) and (not np.min(np.array(geo2)) <= -10000.0):
              part_geos.append(part)
              if part in obj['part_count_occluded'].keys():
                count += obj['part_count_occluded'][part]
              else:
                if part in obj['part_color_occluded'].keys():
                  count += 1

    # if count == 0:
    #   return '__INVALID__'
    # else:
    if "count" in attribute:
      if count >= 10: return '__INVALID__'
      else: return count
    elif "exist" in attribute:
      if count > 0: return True
      else: return False
    else:
      if len(part_geos) != 1 or (not part_geos[0] in obj['part_color_occluded'].keys()): return '__INVALID__'
      else:
        return obj['part_color_occluded'][part_geos[0]][0]
  return filter_handler

def geometry_handler(attribute):
  def filter_handler(scene_struct, inputs, side_inputs):
    if len(inputs) == 1: return inputs[0][2] == attribute
    else: return inputs[2] == attribute
  return filter_handler

def exist_handler(scene_struct, inputs, side_inputs):
  assert len(inputs) == 1
  assert len(side_inputs) == 0
  return len(inputs[0]) > 0


def equal_handler(scene_struct, inputs, side_inputs):
  assert len(inputs) == 2
  assert len(side_inputs) == 0
  return inputs[0] == inputs[1]


def less_than_handler(scene_struct, inputs, side_inputs):
  assert len(inputs) == 2
  assert len(side_inputs) == 0
  return inputs[0] < inputs[1]


def greater_than_handler(scene_struct, inputs, side_inputs):
  assert len(inputs) == 2
  assert len(side_inputs) == 0
  return inputs[0] > inputs[1]

def sum_handler(scene_struct, inputs, side_inputs):
  assert len(inputs) == 2
  assert len(side_inputs) == 0
  if inputs[0] + inputs[1] >= 10: return '__INVALID__'
  return inputs[0] + inputs[1]

def minus_more_handler(scene_struct, inputs, side_inputs):
  assert len(inputs) == 2
  assert len(side_inputs) == 0
  if inputs[0] < inputs[1]: return '__INVALID__'
  return inputs[0] - inputs[1]

def minus_less_handler(scene_struct, inputs, side_inputs):
  assert len(inputs) == 2
  assert len(side_inputs) == 0
  if inputs[0] > inputs[1]: return '__INVALID__'
  return inputs[1] - inputs[0]

execute_handlers = {
  'scene': scene_handler,
  #filter
  'filter_color': make_filter_handler('Color'),
  'filter_object-category': make_filter_handler('Object-Category'),
  'filter_part-count': make_filter_handler('Part-Count'),
  'filter_part-category': make_filter_handler('Part-Category'),
  'filter_stability': make_physics_filter_handler('stability'),
  'filter_unstability': make_physics_filter_handler('unstability'),
  
  #query object
  'query_object-category': make_query_handler('Object-Category'),
  'query_stability': make_physics_query_handler('stability'),
  'query_unstability': make_physics_query_handler('unstability'),
  'query_direction': make_physics_query_handler('direction'),
  'query_change': make_physics_query_handler('change'),
  'query_part-color': make_part_query_handler('Color'),
  'query_part-count': make_part_query_handler('Part-Count'),
  'query_part-category': make_part_query_handler('Part-Category'),
  'query_part-geometry': make_part_query_handler('Part-Geometry'),
  #same
  'same_object-category': make_same_attr_handler('Object-Category'),
  'same_part-color': make_same_part_attr_handler('part_color_occluded'),
  'same_part-count': make_same_part_attr_handler('part_count_occluded'),

  'unique': unique_handler,
  'relate': relate_handler,
  'union': union_handler,
  'intersect': intersect_handler,
  'count': count_handler,
  'exist': exist_handler,

  #comparison
  'equal_part-color': equal_handler,
  'equal_part-count': equal_handler,
  'equal_integer': equal_handler,
  'equal_part-category': equal_handler,
  'equal_object-category': equal_handler,
  'equal_object': equal_handler,
  'less_than': less_than_handler,
  'greater_than': greater_than_handler,

  #arithmetic
  'sum': sum_handler,
  'minus_more': minus_more_handler,
  'minus_less': minus_less_handler,
  
  'count_perpendicular_line': make_filter_geometry_handler('count_perpendicular-line'),
  'count_parallel_line': make_filter_geometry_handler('count_parallel-line'),
  'count_perpendicular_plane': make_filter_geometry_handler('count_perpendicular-plane'),
  'count_parallel_plane': make_filter_geometry_handler('count_parallel-plane'),
  'exist_perpendicular_line': make_filter_geometry_handler('exist_perpendicular-line'),
  'exist_parallel_line': make_filter_geometry_handler('exist_parallel-line'),
  'exist_perpendicular_plane': make_filter_geometry_handler('exist_perpendicular-plane'),
  'exist_parallel_plane': make_filter_geometry_handler('exist_parallel-plane'),
  'query_perpendicular_line_color': make_filter_geometry_handler('query_perpendicular_line_color'),
  'query_parallel_line_color': make_filter_geometry_handler('query_parallel_line_color'),
  'query_perpendicular_plane_color': make_filter_geometry_handler('query_perpendicular_plane_color'),
  'query_parallel_plane_color': make_filter_geometry_handler('query_parallel_plane_color'),
  'query_geometric-relation': query_geometric_relation_handler,
  'query_geometric-analogy-color': query_geometric_analogy_handler('query_geometric-analogy-color'),
  'query_geometric-analogy-count': query_geometric_analogy_handler('query_geometric-analogy-count'),
  'query_positional-relation': query_position_handler,
  'query_positional-analogy-category': query_position_analogy_handler('query_positional-analogy-color'),
  'query_positional-analogy-count': query_position_analogy_handler('query_positional-analogy-count'),
  'query_positional-analogy-exist': query_position_analogy_handler('query_positional-analogy-exist'),
  'perpendicular': geometry_handler('perpendicular'),
  'parallel': geometry_handler('parallel'),
}


def answer_question(question, metadata, scene_struct, all_outputs=False,
                    cache_outputs=True):
  """
  Use structured scene information to answer a structured question. Most of the
  heavy lifting is done by the execute handlers defined above.

  We cache node outputs in the node itself; this gives a nontrivial speedup
  when we want to answer many questions that share nodes on the same scene
  (such as during question-generation DFS). This will NOT work if the same
  nodes are executed on different scenes.
  """
  all_input_types, all_output_types = [], []
  node_outputs = []

  output1 = False
  output2 = False
  for node in question['nodes']:
    if cache_outputs and '_output' in node:
      node_output = node['_output']
    else:
      
      node_type = node['type']
      msg = 'Could not find handler for "%s"' % node_type
      assert node_type in execute_handlers, msg
      handler = execute_handlers[node_type]
      node_inputs = [node_outputs[idx] if isinstance (idx, int) else idx for idx in node['inputs']]
      side_inputs = node.get('side_inputs', [])
      node_output = handler(scene_struct, node_inputs, side_inputs)
      if cache_outputs:
        node['_output'] = node_output
      if node['type'] == "filter_object-category": output1 = node_output
      if node['type'] in ["filter_part-count", "filter_part-category", "filter_color"]: output2 = node_output
    node_outputs.append(node_output)
    if node_output == '__INVALID__':
      break

  if output1 and output2 and output1 == output2: node_outputs[-1] = '__INVALID__'

  if all_outputs:
    return node_outputs
  else:
    return node_outputs[-1]


def insert_scene_node(nodes, idx):
  # First make a shallow-ish copy of the input
  new_nodes = []
  for node in nodes:
    new_node = {
      'type': node['type'],
      'inputs': node['inputs'],
    }
    if 'side_inputs' in node:
      new_node['side_inputs'] = node['side_inputs']
    new_nodes.append(new_node)

  # Replace the specified index with a scene node
  new_nodes[idx] = {'type': 'scene', 'inputs': []}

  # Search backwards from the last node to see which nodes are actually used
  output_used = [False] * len(new_nodes)
  idxs_to_check = [len(new_nodes) - 1]
  while idxs_to_check:
    cur_idx = idxs_to_check.pop()
    output_used[cur_idx] = True
    idxs_to_check.extend(new_nodes[cur_idx]['inputs'])

  # Iterate through nodes, keeping only those whose output is used;
  # at the same time build up a mapping from old idxs to new idxs
  old_idx_to_new_idx = {}
  new_nodes_trimmed = []
  for old_idx, node in enumerate(new_nodes):
    if output_used[old_idx]:
      new_idx = len(new_nodes_trimmed)
      new_nodes_trimmed.append(node)
      old_idx_to_new_idx[old_idx] = new_idx

  # Finally go through the list of trimmed nodes and change the inputs
  for node in new_nodes_trimmed:
    new_inputs = []
    for old_idx in node['inputs']:
      new_inputs.append(old_idx_to_new_idx[old_idx])
    node['inputs'] = new_inputs

  return new_nodes_trimmed


def is_degenerate(question, metadata, scene_struct, answer=None, verbose=False):
  """
  A question is degenerate if replacing any of its relate nodes with a scene
  node results in a question with the same answer.
  """
  if answer is None:
    answer = answer_question(question, metadata, scene_struct)

  for idx, node in enumerate(question['nodes']):
    if 'relate' in node['type']:
      new_question = {
        'nodes': insert_scene_node(question['nodes'], idx)
      }
      new_answer = answer_question(new_question, metadata, scene_struct)
      # if verbose:
      # print('here is truncated question:')
      # for i, n in enumerate(new_question['nodes']):
      #   name = n['type']
      #   if 'side_inputs' in n:
      #     name = '%s[%s]' % (name, n['side_inputs'][0])
      #   # print(i, name, n['_output'])
      # print('new answer is: ', new_answer)

      if new_answer == answer:
        return True

  return False

