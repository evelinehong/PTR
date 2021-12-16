import sys, random, os
import bpy, bpy_extras

# """
# Some utility functions for interacting with Blender
# """
import numpy as np
import math
from mathutils import Matrix

def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')

    last_elem = 0
    running_length = 0

    for i, elem in enumerate(binary_mask.ravel(order='F')):
        if elem == last_elem:
            pass
        else:
            counts.append(running_length)
            running_length = 0
            last_elem = elem
        running_length += 1

    counts.append(running_length)

    return rle
    
def create_sub_mask_annotation(sub_mask):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    contours = measure.find_contours(sub_mask, 0.5, positive_orientation='low')

    segmentations = []
    polygons = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)
        polygons.append(poly)
        segmentation = np.array(poly.exterior.coords).ravel().tolist()
        segmentations.append(segmentation)

    # Combine the polygons to calculate the bounding box and area
    multi_poly = MultiPolygon(polygons)
    x, y, max_x, max_y = multi_poly.bounds
    width = max_x - x
    height = max_y - y
    bbox = (x, y, width, height)
    area = multi_poly.area

    # annotation = {
    #     'segmentation': segmentations,
    #     'iscrowd': is_crowd,
    #     'image_id': image_id,
    #     'category_id': category_id,
    #     'id': annotation_id,
    #     'bbox': bbox,
    #     'area': area
    # }

    return segmentations, bbox, area

def extract_args(input_argv=None):
  """
  Pull out command-line arguments after "--". Blender ignores command-line flags
  after --, so this lets us forward command line arguments from the blender
  invocation to our own script.
  """
  if input_argv is None:
    input_argv = sys.argv
  output_argv = []
  if '--' in input_argv:
    idx = input_argv.index('--')
    output_argv = input_argv[(idx + 1):]
  return output_argv


def parse_args(parser, argv=None):
  return parser.parse_args(extract_args(argv))


# I wonder if there's a better way to do this?
def delete_object(obj):
  """ Delete a specified blender object """
  for o in bpy.data.objects:
    o.select = False
  obj.select = True
  bpy.ops.object.delete()


def get_camera_coords(cam, pos):
  """
  For a specified point, get both the 3D coordinates and 2D pixel-space
  coordinates of the point from the perspective of the camera.

  Inputs:
  - cam: Camera object
  - pos: Vector giving 3D world-space position

  Returns a tuple of:
  - (px, py, pz): px and py give 2D image-space coordinates; pz gives depth
    in the range [-1, 1]
  """
  scene = bpy.context.scene
  x, y, z = bpy_extras.object_utils.world_to_camera_view(scene, cam, pos)
  scale = scene.render.resolution_percentage / 100.0 
  w = int(scale * scene.render.resolution_x)
  h = int(scale * scene.render.resolution_y)
  px = int(round(x * w))
  py = int(round(h - y * h))
  return (px, py, z)


def set_layer(obj, layer_idx):
  """ Move an object to a particular layer """
  # Set the target layer to True first because an object must always be on
  # at least one layer.
  obj.layers[layer_idx] = True
  for i in range(len(obj.layers)):
    obj.layers[i] = (i == layer_idx)

def add_object(name, loc, cur_render_dir, theta=0):
  """
  Load an object from a file. We assume that in the directory object_dir, there
  is a file named "$name.blend" which contains a single object named "$name"
  that has unit size and is centered at the origin.

  - scale: scalar giving the size that the object should be in the scene
  - loc: tuple (x, y) giving the coordinates on the ground plane where the
    object should be placed.
  """
  files = []
  with bpy.data.libraries.load("%s/"%cur_render_dir+name+".blend") as (data_from, data_to):
    for n in data_from.objects:
      if name in n:
        files.append({'name': n})

  bpy.ops.wm.append(directory="%s/"%cur_render_dir+name+".blend"+"/Object/", files = files)

  # Set the new object as active, then rotate, and translate it
  x, y = loc

  ctx = bpy.context.copy()
  obs = []
  
  bpy.context.scene.use_nodes = True
  tree = bpy.context.scene.node_tree

  # prev = bpy.context.area.type
  # bpy.context.area.type = 'NODE_EDITOR'

  # area = bpy.context.area    

  # clear default nodes
  # for node in tree.nodes:
  #         tree.nodes.remove(node)

  render = tree.nodes['Render Layers']
  links = tree.links

  output_node = bpy.context.scene.node_tree.nodes.new('CompositorNodeOutputFile')
  output_node.base_path = cur_render_dir
  # link = links.new(render.outputs["Image"], output_node.inputs[0])

  i = 0

  minz = 100000.0
  for obj in bpy.data.objects:
    if name in obj.name:
      mx = obj.matrix_world
      minz = min(min((mx * v.co)[2] for v in obj.data.vertices), minz)

  # mask_node = bpy.context.scene.node_tree.nodes.new('CompositorNodeIDMask')
  # mask_node.index = 0
  # link = links.new(render.outputs["IndexOB"], mask_node.inputs["ID value"])
  # link = links.new(mask_node.outputs[0], output_node.inputs[0])

  for obj in bpy.data.objects:
    if name in obj.name:
      obj.rotation_euler[2] = theta
      obj.location = (x, y, -minz+0.02)
      # bpy.ops.transform.translate(value=(x, y, -minz))
      # mx = obj.matrix_world
      # mx.translation.z -= minz
      i += 1
      obj.pass_index = i
      mask_node = bpy.context.scene.node_tree.nodes.new('CompositorNodeIDMask')
      mask_node.index = i
      link = links.new(render.outputs["IndexOB"], mask_node.inputs["ID value"])
      output_node.layer_slots.new(str(i))
      link = links.new(mask_node.outputs[0], output_node.inputs[i])
      obs.append(obj)
      ctx['active_object'] = obj
    else:
      obj.hide_render = True
  
  bpy.ops.render.render()
      
  # bpy.context.area.type = prev
  for node in bpy.context.scene.node_tree.nodes:
    if node.name == "Render Layers": continue
    bpy.context.scene.node_tree.nodes.remove(node)

  ctx['selected_objects'] = obs
  bpy.ops.object.join(ctx)

  rotation = []
  location = []

  for obj in bpy.data.objects:
    obj.hide_render = False
    if name in obj.name:
      obj.name = name
      obj.pass_index = 0
      rotation = obj.rotation_euler
      location = obj.location
      # mx = obj.matrix_world
      # minz = min((mx * v.co)[2] for v in obj.data.vertices)
      # mx.translation.z -= minz

  bpy.context.scene.objects.active = bpy.data.objects[name]
  
  return location, rotation

def add_object2(name, loc, rot1, normals, tmp_dir, theta=0.0, mode="ground"):
  rot0 = np.array([  [1.0000000,  0.0000000,  0.0000000],
   [0.0000000,  0.0000000, -1.0000000],
   [0.0000000,  1.0000000,  0.0000000 ]])

  sr = random.random()
  if 'Cart' in name or 'Refrigerator' in name:
    sr = 1.01

  if mode == 'support' and sr < 0.7:
    tilt0 = [[[  1.0000000,  0.0000000,  0.0000000],
        [0.0000000, -1.0000000, 0.0000000],
        [0.0000000,   0.0000, -1.0000000]],
        [[  0.0000000,  1.0000000,  0.0000000],
        [1.0000000, 0.0000000, 0.0000000],
        [0.0000000,   0.0000, -1.0000000]],
        [[  0.0000000,  -1.0000000,  0.0000000],
        [-1.0000000, 0.0000000, 0.0000000],
        [0.0000000,   0.0000, -1.0000000]]
        ]
    rot1 = random.choice(tilt0)
  """
  Load an object from a file. We assume that in the directory object_dir, there
  is a file named "$name.blend" which contains a single object named "$name"
  that has unit size and is centered at the origin.

  - scale: scalar giving the size that the object should be in the scene
  - loc: tuple (x, y) giving the coordinates on the ground plane where the
    object should be placed.
  """
  import math
  def rotation_matrix_from_vectors(vec1, vec2):
    
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

  def isclose(x, y, rtol=1.e-5, atol=1.e-8):
    return abs(x-y) <= atol + rtol * abs(y)

  def euler_angles_from_rotation_matrix(R):
    '''
    From a paper by Gregory G. Slabaugh (undated),
    "Computing Euler angles from a rotation matrix
    '''
    phi = 0.0
    if isclose(R[2,0],-1.0):
        theta = math.pi/2.0
        psi = math.atan2(R[0,1],R[0,2])
    elif isclose(R[2,0],1.0):
        theta = -math.pi/2.0
        psi = math.atan2(-R[0,1],-R[0,2])
    else:
        theta = -math.asin(R[2,0])
        cos_theta = math.cos(theta)
        psi = math.atan2(R[2,1]/cos_theta, R[2,2]/cos_theta)
        phi = math.atan2(R[1,0]/cos_theta, R[0,0]/cos_theta)
    return psi, theta, phi

  import copy

  rot1_copy = rot1.copy()
  rot1 = np.array(rot1)

  tilt = [[[  1.0000000,  0.0000000,  0.0000000],
   [0.0000000, -1.0000000, 0.0000000],
   [0.0000000,   0.0000, -1.0000000]],
   [[ -1.0000000,  0.0000000,  0.0000000],
   [0.0000000,  1.0000000,  0.0000000],
  [-0.0000000,  0.0000000, -1.0000000 ]]]

  rot1 = np.array(rot1) @ rot0

  if mode == "side wall" or mode == "back wall":
    if random.random() < 0.75:
      theta = 0.0

  rot3 = Rz(theta)


  files = []
  with bpy.data.libraries.load("%s/"%tmp_dir+name+".blend") as (data_from, data_to):
    for n in data_from.objects:
      if name in n:
        files.append({'name': n})

  bpy.ops.wm.append(directory="%s/"%tmp_dir+name+".blend"+"/Object/", files = files)

  # Set the new object as active, then rotate, and translate it
  x, y = loc

  ctx = bpy.context.copy()
  obs = []
  
  bpy.context.scene.use_nodes = True
  tree = bpy.context.scene.node_tree

  # prev = bpy.context.area.type
  # bpy.context.area.type = 'NODE_EDITOR'

  # area = bpy.context.area    

  # clear default nodes
  # for node in tree.nodes:
  #         tree.nodes.remove(node)

  render = tree.nodes['Render Layers']
  links = tree.links

  output_node = bpy.context.scene.node_tree.nodes.new('CompositorNodeOutputFile')
  output_node.base_path = tmp_dir

  # link = links.new(render.outputs["Image"], output_node.inputs[0])

  minz = 100000.0
  first_min = []
  second_min = []
  if 'Bed' in name or 'Chair' in name:
    if rot1_copy == tilt[0]:
      miny = -100000.0
    if rot1_copy == tilt[1]:
      miny = 100000.0

  names = []
  normal = []
  for obj in bpy.data.objects:
    if name in obj.name:
      names.append(obj.name)

  maxz = -10000.0
  maxx = -10000.0
  minx = 10000.0
  maxy = -10000.0
  miny = 10000.0

  for obj in bpy.data.objects:
    if name in obj.name:
      obj.rotation_euler = euler_angles_from_rotation_matrix(rot1)
      bpy.context.scene.update()
      mx = obj.matrix_world
      
      if ('Bed' in name or 'Chair' in name) and (rot1_copy in tilt) and not (mode == 'support' and sr < 0.7):
          
        if "headboard" in obj.name or "back" in obj.name:
          mz, idx = min([(mx * v.co)[2], idx] for idx, v in enumerate(obj.data.vertices))
          first_min = [(mx * v.co) for v in obj.data.vertices][idx]

      if ('Bed' in name or 'Chair' in name) and not (mode == 'support' and sr < 0.7):
        if rot1_copy == tilt[0]:
          
          if "seat" in obj.name or "sleep" in obj.name:
            my, idx = max([(mx * v.co)[1], idx] for idx, v in enumerate(obj.data.vertices))
            second_min = [(mx * v.co) for v in obj.data.vertices][idx]

        if rot1_copy == tilt[1]:
          if "seat" in obj.name or "sleep" in obj.name:
            my, idx = min([(mx * v.co)[1], idx] for idx, v in enumerate(obj.data.vertices))
            second_min = [(mx * v.co) for v in obj.data.vertices][idx]

      if mode == 'normal':
        if "top" in obj.name or "sleep" in obj.name or "seat" in obj.name or "arm" in obj.name or 'Refrigerator' in name:
          maxz = max(max((mx * v.co)[2] for v in obj.data.vertices), maxz)
          maxx = max(max((mx * v.co)[0] for v in obj.data.vertices), maxz)
          minx = min(min((mx * v.co)[0] for v in obj.data.vertices), minz)
          maxy = max(max((mx * v.co)[1] for v in obj.data.vertices), maxz)
          miny = min(min((mx * v.co)[1] for v in obj.data.vertices), minz)

          normal = [maxz, maxx, minx, maxy, miny]

  if ('Bed' in name or 'Chair' in name) and (rot1_copy in tilt) and len(first_min) and len(second_min) and not (mode == 'support' and sr < 0.7):
    first_min[0] = 0
    second_min[0] = 0
    
    dist = math.sqrt((second_min[1] - first_min[1]) ** 2 + (second_min[2] - first_min[2]) ** 2)
    new=[0,0,first_min[2]]
    if second_min[1] > first_min[1]:
      new[1] = first_min[1] + dist
    else:
      new[1] = first_min[1] - dist

    rot2 = rotation_matrix_from_vectors(np.array(second_min - first_min).tolist(), (np.array(new) - np.array(first_min)).tolist())

    rot = np.array(rot3) @ np.array(rot2) @ np.array(rot1)

  else:
    rot = np.array(rot3) @ np.array(rot1)

  if mode == "side wall":
    side = random.random()
    if side < 0.5:
      ry = Ry(random.random() + 0.45)
      rot = np.array(ry) @ rot
    else:
      ry = Ry(-random.random() + 0.45)
      rot = np.array(ry) @ rot

  if mode == "back wall":
    rx = Rx(-random.random() + 0.45)
    rot = np.array(rx) @ rot

    side = random.random()
    if side < 0.5:
      side2 = random.random()
      if side2 < 0.5:
        ry = Ry(random.random() + 0.45)
        rot = np.array(ry) @ rot
      else:
        ry = Ry(-random.random() + 0.45)
        rot = np.array(ry) @ rot
 
  minz = 10000.0
  maxx = -10000.0
  minx = 10000.0
  maxy = -10000.0
  miny = 10000.0
  minzz = 10000.0
  for obj in bpy.data.objects:
    if name in obj.name: 
      obj.rotation_euler = euler_angles_from_rotation_matrix(rot)
      bpy.context.scene.update()
      mx = obj.matrix_world     
      minz = min(min((mx * v.co)[2] for v in obj.data.vertices), minz)
      maxx = max(max((mx * v.co)[0] for v in obj.data.vertices), maxx)
      maxy = max(max((mx * v.co)[1] for v in obj.data.vertices), maxy)
      miny = min(min((mx * v.co)[1] for v in obj.data.vertices), miny)
      minx = min(min((mx * v.co)[0] for v in obj.data.vertices), minx)

      if mode == 'support' and sr < 0.7:
        if "top" in obj.name or "sleep" in obj.name or "seat" in obj.name or "arm" in obj.name:
          minzz = min(min((mx * v.co)[2] for v in obj.data.vertices), minzz)

  if mode == "side wall":
    if random.random() < 0.5:
      x = - maxx + 4.99
    else:
      x = - minx - 4.99

  if mode == "back wall":
    y = -maxy + 2.99
    if side < 0.5:
      if random.random() < 0.5:
        x = - maxx + 4.99
      else:
        x = - minx - 4.99
  
  if mode == 'support':
    
    z = max(-minz + normals[-1][-1][0] + normals[-1][-2][2] + 0.01, -minz + 0.02)
    if minzz != 10000.0 and sr < 0.7:
      z = max(-minzz + normals[-1][-1][0] + normals[-1][-2][2] + 0.01, -minz + 0.02)

    y = random.uniform(-miny + normals[-1][-1][-1], -maxy + normals[-1][-1][-2])
    x = random.uniform(-minx + normals[-1][-1][-3], -maxx + normals[-1][-1][1])

    if sr < 0.7:
      if rot1_copy == tilt0[0]:
        y = -miny + normals[-1][-1][-1] + random.random() * (maxy - miny) / 1.25 -  (maxy - miny)/1.1
        x = random.uniform(-minx + normals[-1][-1][-3], -maxx + normals[-1][-1][1])

      if rot1_copy == tilt0[1]:
        x = -minx + normals[-1][-1][-3] + random.random() * (maxx - minx) / 1.25 -  (maxx - minx) / 1.1
        y = random.uniform(-miny + normals[-1][-1][-1], -maxy + normals[-1][-1][-2])

      if rot1_copy == tilt0[2]:
        x = -maxx + normals[-1][-1][1] - random.random() * (maxx - minx) / 1.25 +  (maxx - minx) / 1.1
        y = random.uniform(-miny + normals[-1][-1][-1], -maxy + normals[-1][-1][-2])

  i = 0
  for obj in bpy.data.objects:
    if name in obj.name:   
         
      if mode == 'support':
        obj.location = (x+normals[-1][-2][0],y+normals[-1][-2][1],z) 
      else:
        obj.location = (x, y, -minz+0.02) 

      if mode == "normal":
        bpy.context.scene.update()
        rotation = obj.rotation_euler
        location = obj.location
      # bpy.ops.transform.translate(value=(x, y, -minz))
      # mx = obj.matrix_world
      # mx.translation.z -= minz
      i += 1
      obj.pass_index = i
      mask_node = bpy.context.scene.node_tree.nodes.new('CompositorNodeIDMask')
      mask_node.index = i
      link = links.new(render.outputs["IndexOB"], mask_node.inputs["ID value"])
      output_node.layer_slots.new(str(i))
      link = links.new(mask_node.outputs[0], output_node.inputs[i])
      obs.append(obj)
      ctx['active_object'] = obj
    else:
      if len(normals) and not normals[-1][0] in obj.name:
        obj.hide_render = True

  bpy.ops.render.render()
      
  # bpy.context.area.type = prev
  for node in bpy.context.scene.node_tree.nodes:
    if node.name == "Render Layers": continue
    bpy.context.scene.node_tree.nodes.remove(node)

  if not mode == "normal":
    ctx['selected_objects'] = obs
    bpy.ops.object.join(ctx)
  
  if mode == "support":
    obs2 = []
    os.mkdir("%s_normal"%tmp_dir)
    output_node2 = bpy.context.scene.node_tree.nodes.new('CompositorNodeOutputFile')
    output_node2.base_path = "%s_normal"%tmp_dir

    k = 0
    for obj in bpy.data.objects:      
      if normals[-1][0] in obj.name:   
        print (obj.name)
        obj.hide_render = False
        k += 1
        i += 1
        obj.pass_index = i
        mask_node = bpy.context.scene.node_tree.nodes.new('CompositorNodeIDMask')
        mask_node.index = i
        link = links.new(render.outputs["IndexOB"], mask_node.inputs["ID value"])
        output_node2.layer_slots.new(str(k))
        link = links.new(mask_node.outputs[0], output_node2.inputs[k])
        obs2.append(obj)
      else:
        if not name in obj.name:
          obj.hide_render = True

    bpy.ops.render.render()

    for node in bpy.context.scene.node_tree.nodes:
      if node.name == "Render Layers": continue
      bpy.context.scene.node_tree.nodes.remove(node)

    for obj in bpy.data.objects:
      if normals[-1][0] in obj.name:
        obj.name = normals[-1][0]
        obj.pass_index = 0

    # ctx['selected_objects'] = obs2
    # bpy.ops.object.join(ctx)

  import bmesh
  from mathutils.bvhtree import BVHTree

  valid = True

  if not mode == "normal":
    for obj in bpy.data.objects:
      obj.hide_render = False
      if name in obj.name:
        obj.name = name
        obj.pass_index = 0
        rotation = obj.rotation_euler
        location = obj.location

        if mode == "support":
          bm1 = bmesh.new()
          bm1.from_mesh(obj.data)
          bm1.transform(bpy.context.scene.objects[name].matrix_world)
          tree1 = BVHTree.FromBMesh(bm1)

          for obj2 in bpy.data.objects:
            if obj.name == obj2.name: continue
            if 'Table' in obj2.name or 'Chair' in obj2.name or 'Refrigerator' in obj2.name or 'Cart' in obj2.name or 'Bed' in obj2.name: 
              name2 = obj2.name
              bm2 = bmesh.new()

              #fill bmesh data from objects
              
              bm2.from_mesh(obj2.data) 
              bm2.transform(bpy.context.scene.objects[name2].matrix_world)
              tree2 = BVHTree.FromBMesh(bm2)

              inter = tree1.overlap(tree2)

              if len(inter):
                valid = False
      # mx = obj.matrix_world
      # minz = min((mx * v.co)[2] for v in obj.data.vertices)
      # mx.translation.z -= minz

  if not mode == "normal":
    bpy.context.scene.objects.active = bpy.data.objects[name]
  
  return location, rotation, normal, valid

def Rx(theta):
  return np.matrix([[ 1, 0           , 0           ],
                  [ 0, math.cos(theta),-math.sin(theta)],
                  [ 0, math.sin(theta), math.cos(theta)]])
  
def Ry(theta):
  return np.matrix([[ math.cos(theta), 0, math.sin(theta)],
                  [ 0           , 1, 0           ],
                  [-math.sin(theta), 0, math.cos(theta)]])
  
def Rz(theta):
  return np.matrix([[ math.cos(theta), -math.sin(theta), 0 ],
                  [ math.sin(theta), math.cos(theta) , 0 ],
                  [ 0           , 0            , 1 ]])


def revise_annotations(line_geo, plane_geo, part_color, part_count, all_objects, obj_name, part_list2, count_list, theta):
    rotation_matrix = np.array(((np.cos(theta), -np.sin(theta), 0),
                (np.sin(theta),  np.cos(theta), 0 ),
                (0, 0, 1))) @ np.array([  [1.0000000,  0.0000000,  0.0000000],
                [0.0000000,  0.0000000, -1.0000000],
                [0.0000000,  1.0000000,  0.0000000 ]])

    line_geo_final = dict(); plane_geo_final = dict(); part_color_all = dict(); part_color_final = dict(); part_count_final = dict(); final_objects = []

    geometry = True
    for part, g in line_geo.items():
      part = rename_part(part, obj_name)
      stand, geometry = check_g(g)

      geo = [stand[0], stand[1], stand[2]]
      geo = rotation_matrix.dot(geo).tolist()

      line_geo_final[part] = geo
    
    for part, g in plane_geo.items():
      part = rename_part(part, obj_name)
      stand = g[0]
      geo = [stand[0], stand[1], stand[2]]
      geo = rotation_matrix.dot(geo).tolist()

      plane_geo_final[part] = geo

    for part, color in part_color.items():
      part_color_all[part] = color
      part = rename_part(part, obj_name)
      
      if part in part_list2:
        part_color_final[part] = color
    
    for part in all_objects:
      part = rename_part(part, obj_name)
      final_objects.append(part)

    for part, count in part_count.items():
      part = rename_part(part, obj_name)
      if part in count_list:
        if not part in part_count_final.keys():
          part_count_final[part] = count
        else:
          part_count_final[part] += count

    return line_geo_final, plane_geo_final, part_color_all, part_color_final, part_count_final, geometry, final_objects

def check_part(obj_name, part_count_final, part_color_final):
    keep = True
    if "wheel" in part_count_final.keys() and obj_name in ['Chair', 'Table']:
      if "leg" in part_count_final.keys():
        part_count_final["wheel"] = part_count_final["leg"]
      else:
        print ("wheel not paired with leg")
        keep = False
        
    if obj_name == 'Chair' and not ('leg' in part_color_final.keys() or 'central_support' in part_color_final.keys() or 'pedestal' in part_color_final.keys()):
      print ("lack base of chair")
      keep = False

    if obj_name == 'Refrigerator' and not 'door' in part_color_final.keys():
      print ("lack door of fridge")
      keep = False

    if obj_name == 'Chair' and ('arm' in part_color_final.keys() and ('arm vertical bar' in part_color_final.keys() or 'arm horizontal bar' in part_color_final.keys())):
      print ("duplicate arm entry")
      keep = False
    return keep

def check_g(g):
  geometry = True
  stand = g[0]
  if len(g) > 1:
    invalids = 0
    stand = [10000,10000,10000]
    for geo in g:
      if geo[0] == 10000:
        invalids += 1
      else:
        if not stand == [10000,10000,10000]:
          stand = geo

    if invalids >= 2 and invalids <= len(g) - 2:
      geometry = False
    else:
      if invalids < 2:
        invalids2 = 0
        for geo in g:
          if np.linalg.norm(np.array(geo) - np.array(stand)) > 0.3: invalids2 += 1 
        if invalids2 >= 2:
          geometry = False

      elif invalids > len(g) - 2:
        stand = [10000,10000,10000]

  return stand, geometry   

def get_rot(obj_name, mode, p_type):
  if obj_name in ['Chair']:
    tilt = [[[  1.0000000,  0.0000000,  0.0000000],
    [0.0000000, -1.0000000, 0.0000000],
    [0.0000000,   0.0000, -1.0000000]],
    [[ -1.0000000,  0.0000000,  0.0000000],
    [0.0000000,  1.0000000,  0.0000000],
    [-0.0000000,  0.0000000, -1.0000000 ]],
    [[  1.0000000,  0.0000000,  0.0000000],
    [0.0000000, -1.0000000, 0.0000000],
    [0.0000000,   0.0000, -1.0000000]],
    [[  0.0000000,  0.0000000,  1.0000000],
    [0.0000000, 1.0000000, 0.0000000],
    [-1.0000000,   0.0000, 0.0000000]],
    [[  0.0000000,  0.0000000,  -1.0000000],
    [0.0000000, 1.0000000, 0.0000000],
    [1.0000000,   0.0000, 0.0000000]],
    [[  1.0000000,  0.0000000,  0.0000000],
    [0.0000000, 0.0000000, -1.0000000],
    [0.0000000,   1.0000, 0.0000000]],
    [[  1.0000000,  0.0000000,  0.0000000],
    [0.0000000, 0.0000000, -1.0000000],
    [0.0000000,   1.0000, 0.0000000]],
    [[  1.0000000,  0.0000000,  0.0000000],
    [0.0000000, 0.0000000, 1.0000000],
    [0.0000000,   -1.0000, 0.0000000]],
    [[  1.0000000,  0.0000000,  0.0000000],
    [0.0000000, 1.0000000, 0.0000000],
    [0.0000000,   0.0000, 1.0000000]]]
  else:
    tilt = [[[  -1.0000000,  0.0000000,  0.0000000],
      [0.0000000, 1.0000000, 0.0000000],
      [0.0000000,   0.0000, -1.0000000]],
    [[  1.0000000,  0.0000000,  0.0000000],
      [0.0000000, 1.0000000, 0.0000000],
      [0.0000000,   0.0000, 1.0000000]]]


  rot = random.choice(tilt)

  if 'Cart' in obj_name or mode == "normal" or p_type == "physics":
    rot = [[  1.0000000,  0.0000000,  0.0000000],
        [0.0000000, 1.0000000, 0.0000000],
        [0.0000000,   0.0000, 1.0000000]]    

  return rot 

def create_urdf_object(obj_name, obj_name2, tmp_dir, ori, cur_shape_dir, p_type, id2):
    if (not 'Cart' in obj_name) and p_type != "physics":
      obj_files = os.listdir(os.path.join("%s_urdf"%tmp_dir, obj_name2))
      obj_files = [obj_file for obj_file in obj_files if obj_file.endswith(".obj")]
      urdf_file = os.path.join("%s_urdf"%tmp_dir, obj_name2, "mobility.urdf")
      f0 = open(urdf_file, "a")
      f0.write("<?xml version=\"1.0\" ?>\n"+\
              "<robot name=\"partnet_"+id2+"\">\n"+\
              "<link name=\"base\">\n")
      for obj_file in obj_files:
        o = obj_file.replace(".obj", "")
        f0.write("<visual name=\""+o+"\">\n"+\
                              "<origin rpy=\"" + str(ori[0]) + " " + str(ori[1]) + " " + str(ori[2]) + "\" " + "xyz=\"0.0 0.0 0.0\"/>\n"+\
                              "<geometry>\n"+\
                              "<mesh filename=\""+o+".obj\"/>\n"+\
                              "</geometry> \n"+\
                              # "<material name=\"black_metal\"> \n"+\
                              # "<color rgba=\""+str(color[0]) + " " +str(color[1]) + " " +str(color[2]) +" 1\"/>\n" + \
                              # "</material>\n"+\
                              "</visual>\n"+\
                              "<collision>\n"+\
                              "<origin rpy=\"" + str(ori[0]) + " " + str(ori[1]) + " " + str(ori[2]) + "\" " +  "xyz=\"0.0 0.0 0.0\"/>\n"+\
                              "<geometry>\n"+\
                              "<mesh filename=\""+o+".obj\"/>\n"+\
                              "</geometry>\n"+\
                              "</collision>\n")

      f0.write("</link>\n</robot>")
      f0.close()

      urdf_file = os.path.join("tmp_urdf", obj_name2, "mobility.urdf")
    
    else:
      urdf_file = open(os.path.join(cur_shape_dir, "mobility.urdf"))
      urdf = urdf_file.read()
      urdf2 = urdf.replace("  xyz=\"0 0 0\"", " rpy=\"" + str(ori[0]) + " " + str(ori[1]) + " " + str(ori[2]) + "\" xyz=\"0 0 0\"")
      urdf_file2 = open(os.path.join(cur_shape_dir, "mobility_physics.urdf"), "w")
      urdf_file2.write(urdf2)
      urdf_file = os.path.join(cur_shape_dir, "mobility_physics.urdf")

    return urdf_file


def get_list(obj_name):
    if obj_name == 'Chair':
      part_list = ['chair_head', 'chair_back', 'chair_seat', 'leg', 'footrest', 'central_support', 'pedestal', 'leg bar', 'foot', 'mechanical_control', 'caster', 'connector', 'chair_arm', 'arm_near_vertical_bar', 'arm_horizontal_bar']
      part_list2 = ['arm', 'leg', 'back', 'seat', 'central support', 'pedestal', 'leg bar', 'wheel', 'arm vertical bar', 'arm horizontal bar']
      count_list = ['arm', 'leg', 'leg bar', 'wheel', 'arm vertical bar', 'arm horizontal bar', 'central support', 'seat', 'back']
      geo_list1 = ['arm_near_vertical_bar', 'arm_horizontal_bar', 'leg', 'leg bar', 'central_support']
      geo_list2 = ['chair_back', 'chair_seat', 'pedestal']

    if obj_name == 'Refrigerator':
      part_list = ['door_frame', 'frame', 'body_interior', 'base', 'handle']
      part_list2 = ['door', 'body']
      count_list = ['door']
      geo_list1 = []
      geo_list2 = ['door_frame']

    if obj_name == 'Bed':
      part_list = ['bed_sleep_area', 'headboard', 'surface_base', 'leg', 'bar_stretcher']
      part_list2 = ['sleep area', 'back', 'leg']
      count_list = ['leg', 'leg bar']
      geo_list1 = ['leg', 'leg bar']
      geo_list2 = ['headboard']

    if obj_name == 'Table':
      part_list = ['tabletop', 'leg', 'central_support', 'pedestal', 'foot',  'leg bar', 'cabinet_door_surface', 'handle',\
        'shelf', 'drawer_box', 'keyboard_tray', 'foot', 'caster', 'frame']      
      count_list = ['drawer', 'leg', 'door', 'leg bar', 'shelf', 'wheel', 'central support', 'top']
      part_list2 = ['top', 'drawer', 'door', 'central support', 'leg', 'pedestal', 'shelf', 'leg bar', 'wheel']
      geo_list1 = ['leg', 'central_support', 'leg bar']
      geo_list2 = ['pedestal', 'cabinet_door_surface', 'drawer_box', 'tabletop', 'shelf']

    if obj_name == 'Cart':
      part_list = ['base_body', 'handle', 'wheel']
      part_list2 = ['body', 'wheel']
      count_list = ['wheel']
      geo_list1 = []
      geo_list2 = []

    return part_list, part_list2, count_list, geo_list1, geo_list2


def rename_part(part, obj_name):
  if part in ['bar_stretcher', 'circular_stretcher', 'runner', 'rocker']:
    part = 'leg bar'
  if part in ['back_panel', 'vertical_side_panel', 'bottom_panel', 'vertical_divider_panel', 'vertical_front_panel']:
    part = 'frame'
  if part in ['drawer_box']: part = 'drawer'
  if part in ['cabinet_door_surface']: part = 'door'
  if part in ['back_surface']: part='chair_back'
  if part in ['door_frame']: part = 'door'
  if part in ['headboard']: part = 'back'
  if part in ['bed_post']: part = 'leg'
  if part in ['surface_base']: part = 'base'
  if part in ['base_body']: part = 'body'
  if part in ['frame'] and 'Refrigerator' in obj_name: part = 'body'
  if part in ['caster']: part = 'wheel'
  if part in ['arm_near_vertical_bar']: part = 'arm vertical bar'
  if part in ['back_frame_vertical_bar']: part = 'back vertical bar'
  if part in ['back_frame_horizontal_bar']: part = 'back horizontal bar'
  part = part.replace("chair_", "").replace("table", "").replace("bed", "").replace("_", " ").replace("bed", "")
  part = part.strip()

  return part

def get_calibration_matrix_K_from_blender(camd):
    f_in_mm = camd.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if (camd.sensor_fit == 'VERTICAL'):
        # the sensor height is fixed (sensor fit is horizontal), 
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio 
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else: # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal), 
        # the sensor height is effectively changed with the pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm

    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = resolution_x_in_px*scale / 2
    v_0 = resolution_y_in_px*scale / 2
    skew = 0 # only use rectangular pixels

    K = Matrix(
        ((alpha_u, skew,    u_0, 0),
        (    0  ,  alpha_v, v_0, 0),
        (    0  ,    0,      1, 0),
        (0,0,0,1)))
    return K

def get_3x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera
    R_bcam2cv = Matrix(
        ((1, 0,  0),
         (0, -1, 0),
         (0, 0, -1)))

    # Transpose since the rotation is object rotation, 
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam * location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam*cam.location
    # Use location from matrix_world to account for constraints:     
    T_world2bcam = -1*R_world2bcam * location

    # Build the coordinate transform matrix from world to computer vision camera
    # NOTE: Use * instead of @ here for older versions of Blender
    # TODO: detect Blender version
    R_world2cv = R_bcam2cv*R_world2bcam
    T_world2cv = R_bcam2cv*T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],),
        [0, 0, 0, 1]
         ))
    return RT


def get_3x4_P_matrix_from_blender(cam):
    K = get_calibration_matrix_K_from_blender(cam.data)
    RT = get_3x4_RT_matrix_from_blender(cam)
    return K*RT, K, RT