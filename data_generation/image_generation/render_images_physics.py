from __future__ import print_function
import math, sys, random, argparse, json, os, tempfile
from datetime import datetime as dt
from collections import Counter
from add_parts import *
import pybullet as p
from PIL import Image
"""
Renders random scenes using Blender, each with with a random number of objects;
each object has a random size, position, color, and shape. Objects will be
nonintersecting but may partially occlude each other. Output images will be
written to disk as PNGs, and we will also write a JSON file for each image with
ground-truth scene information.

This file expects to be run from Blender like this:

blender --background --python render_images.py -- [arguments to this script]
"""

INSIDE_BLENDER = True
try:
  import bpy, bpy_extras
  from mathutils import Vector
except ImportError as e:
  INSIDE_BLENDER = False
if INSIDE_BLENDER:
  # try:
  import utils
  from utils import *

parser = argparse.ArgumentParser()

# Input options
parser.add_argument('--base_scene_blendfile', default='data/base_scene.blend',
    help="Base blender file on which all scenes are based; includes " +
          "ground plane, lights, and camera.")
parser.add_argument('--properties_json', default='data/properties_partnet.json',
    help="JSON file defining objects, materials, sizes, and colors. " +
         "The \"colors\" field maps from CLEVR color names to RGB values; " +
         "The \"sizes\" field maps from CLEVR size names to scalars used to " +
         "rescale object models; the \"materials\" and \"shapes\" fields map " +
         "from CLEVR material and shape names to .blend files in the " +
         "--object_material_dir and --shape_dir directories respectively.")
parser.add_argument('--tmp_dir', default='./tmp9')
parser.add_argument('--material_dir', default='data/materials',
    help="Directory where .blend files for materials are stored")


# Settings for objects
parser.add_argument('--min_objects', default=3, type=int,
    help="The minimum number of objects to place in each scene")
parser.add_argument('--max_objects', default=3, type=int,
    help="The maximum number of objects to place in each scene")
parser.add_argument('--min_dist', default=0.25, type=float,
    help="The minimum allowed distance between object centers")
parser.add_argument('--margin', default=1.5, type=float,
    help="Along all cardinal directions (left, right, front, back), all " +
         "objects will be at least this distance apart. This makes resolving " +
         "spatial relationships slightly less ambiguous.")
parser.add_argument('--margin2', default=8, type=float,
    help="Along all cardinal directions (left, right, front, back), all " +
         "objects will be at least this distance apart. This makes resolving " +
         "spatial relationships slightly less ambiguous.")
parser.add_argument('--min_pixels_per_object', default=200, type=int,
    help="All objects will have at least this many visible pixels in the " +
         "final rendered images; this ensures that no objects are fully " +
         "occluded by other objects.")
parser.add_argument('--max_retries', default=20000, type=int,
    help="The number of times to try placing an object before giving up and " +
         "re-placing all objects in the scene.")

# Output settings
parser.add_argument('--start_idx', default=0, type=int,
    help="The index at which to start for numbering rendered images. Setting " +
         "this to non-zero values allows you to distribute rendering across " +
         "multiple machines and recombine the results later.")
parser.add_argument('--num_images', default=5, type=int,
    help="The number of images to render")
parser.add_argument('--filename_prefix', default='PARTNET2',
    help="This prefix will be prepended to the rendered images and JSON scenes")
parser.add_argument('--split', default='new',
    help="Name of the split for which we are rendering. This will be added to " +
         "the names of rendered images, and will also be stored in the JSON " +
         "scene structure for each image.")
parser.add_argument('--output_image_dir', default='../output/images_physics/',
    help="The directory where output images will be stored. It will be " +
         "created if it does not exist.")
parser.add_argument('--output_scene_dir', default='../output/scenes_physics/',
    help="The directory where output JSON scene structures will be stored. " +
         "It will be created if it does not exist.")
parser.add_argument('--output_depth_dir', default='../output/depths_physics/',
    help="The directory where output JSON scene structures will be stored. " +
         "It will be created if it does not exist.")
parser.add_argument('--output_scene_file', default='../output/partnet_scenes.json',
    help="Path to write a single JSON file containing all scene information")
parser.add_argument('--output_blend_dir', default='output/blendfiles',
    help="The directory where blender scene files will be stored, if the " +
         "user requested that these files be saved using the " +
         "--save_blendfiles flag; in this case it will be created if it does " +
         "not already exist.")
parser.add_argument('--save_blendfiles', type=int, default=0,
    help="Setting --save_blendfiles 1 will cause the blender scene file for " +
         "each generated image to be stored in the directory specified by " +
         "the --output_blend_dir flag. These files are not saved by default " +
         "because they take up ~5-10MB each.")
parser.add_argument('--version', default='1.0',
    help="String to store in the \"version\" field of the generated JSON file")
parser.add_argument('--license',
    default="Creative Commons Attribution (CC-BY 4.0)",
    help="String to store in the \"license\" field of the generated JSON file")
parser.add_argument('--date', default=dt.today().strftime("%m/%d/%Y"),
    help="String to store in the \"date\" field of the generated JSON file; " +
         "defaults to today's date")

# Rendering options
parser.add_argument('--use_gpu', default=1, type=int,
    help="Setting --use_gpu 1 enables GPU-accelerated rendering using CUDA. " +
         "You must have an NVIDIA GPU with the CUDA toolkit installed for " +
         "to work.")
parser.add_argument('--width', default=800, type=int,
    help="The width (in pixels) for the rendered images")
parser.add_argument('--height', default=600, type=int,
    help="The height (in pixels) for the rendered images")
parser.add_argument('--key_light_jitter', default=1.0, type=float,
    help="The magnitude of random jitter to add to the key light position.")
parser.add_argument('--fill_light_jitter', default=1.0, type=float,
    help="The magnitude of random jitter to add to the fill light position.")
parser.add_argument('--back_light_jitter', default=1.0, type=float,
    help="The magnitude of random jitter to add to the back light position.")
parser.add_argument('--camera_jitter', default=0.5, type=float,
    help="The magnitude of random jitter to add to the camera position")
parser.add_argument('--render_num_samples', default=512, type=int,
    help="The number of samples to use when rendering. Larger values will " +
         "result in nicer images but will cause rendering to take longer.")
parser.add_argument('--render_min_bounces', default=8, type=int,
    help="The minimum number of bounces to use for rendering.")
parser.add_argument('--render_max_bounces', default=8, type=int,
    help="The maximum number of bounces to use for rendering.")
parser.add_argument('--render_tile_size', default=256, type=int,
    help="The tile size to use for rendering. This should not affect the " +
         "quality of the rendered image but may affect the speed; CPU-based " +
         "rendering may achieve better performance using smaller tile sizes " +
         "while larger tile sizes may be optimal for GPU-based rendering.")

parser.add_argument('--data_dir', default='./data_v0', type=str)
parser.add_argument('--mobility_dir', default='./cart', type=str)

def main(args):
  num_digits = 6
  prefix = '%s_%s_' % (args.filename_prefix, args.split)
  img_template = '%s%%0%dd.png' % (prefix, num_digits)
  scene_template = '%s%%0%dd.json' % (prefix, num_digits)
  blend_template = '%s%%0%dd.blend' % (prefix, num_digits)
  img_template = os.path.join(args.output_image_dir, img_template)
  scene_template = os.path.join(args.output_scene_dir, scene_template)
  blend_template = os.path.join(args.output_blend_dir, blend_template)

  if not os.path.isdir(args.output_image_dir):
    os.makedirs(args.output_image_dir)
  if not os.path.isdir(args.output_scene_dir):
    os.makedirs(args.output_scene_dir)
  if args.save_blendfiles == 1 and not os.path.isdir(args.output_blend_dir):
    os.makedirs(args.output_blend_dir)
  
  all_scene_paths = []
  for i in range(args.num_images):
    img_path = img_template % (i + args.start_idx)
    scene_path = scene_template % (i + args.start_idx)
    all_scene_paths.append(scene_path)
    blend_path = None
    if args.save_blendfiles == 1:
      blend_path = blend_template % (i + args.start_idx)
    num_objects = random.randint(args.min_objects, args.max_objects)

    splits = ["train", "val", "test"]
    split_prob = random.random()
    if split_prob < 0.14286:
      split = "val"
    elif split_prob < 0.28571:
      split = "test"
    else:
      split = "train"

    render_scene(args,
      num_objects=num_objects,
      output_index=(i + args.start_idx),
      output_split=args.split,
      output_image=img_path,
      output_scene=scene_path,
      output_blendfile=blend_path,
      split = split
    )

  # After rendering all images, combine the JSON files for each scene into a
  # single JSON file.
  all_scenes = []
  for scene_path in all_scene_paths:
    with open(scene_path, 'r') as f:
      all_scenes.append(json.load(f))
  output = {
    'info': {
      'date': args.date,
      'version': args.version,
      'split': args.split,
      'license': args.license,
    },
    'scenes': all_scenes
  }
  with open(args.output_scene_file, 'w') as f:
    json.dump(output, f)

def render_scene(args,
    num_objects=5,
    output_index=0,
    output_split='none',
    output_image='render.png',
    output_scene='render_json',
    output_blendfile=None,
    split='train'
  ):

  os.mkdir("%s_urdf"%args.tmp_dir)

  # Load the main blendfile
  base_scene_blendfiles = ['data/base_scene_physics.blend']
  bpy.ops.wm.open_mainfile(filepath=random.choice(base_scene_blendfiles))


  # Set render arguments so we can get pixel coordinates later.
  # We use functionality specific to the CYCLES renderer so BLENDER_RENDER
  # cannot be used.
  render_args = bpy.context.scene.render
  render_args.engine = "CYCLES"
  render_args.filepath = output_image
  render_args.resolution_x = args.width
  render_args.resolution_y = args.height
  render_args.resolution_percentage = 100
  render_args.tile_x = args.render_tile_size
  render_args.tile_y = args.render_tile_size
  if args.use_gpu == 1:
    # Blender changed the API for enabling CUDA at some point
    if bpy.app.version < (2, 78, 0):
      bpy.context.user_preferences.system.compute_device_type = 'CUDA'
      bpy.context.user_preferences.system.compute_device = 'CUDA_0'
    else:
      cycles_prefs = bpy.context.user_preferences.addons['cycles'].preferences
      cycles_prefs.compute_device_type = 'CUDA'

  # Some CYCLES-specific stuff
  bpy.data.worlds['World'].cycles.sample_as_light = True
  bpy.context.scene.cycles.blur_glossy = 2.0
  bpy.context.scene.cycles.samples = args.render_num_samples
  bpy.context.scene.cycles.transparent_min_bounces = args.render_min_bounces
  bpy.context.scene.cycles.transparent_max_bounces = args.render_max_bounces
  if args.use_gpu == 1:
    bpy.context.scene.cycles.device = 'GPU'

  # This will give ground-truth information about the scene and its objects
  scene_struct = {
      'split': output_split,
      'image_index': output_index,
      'image_filename': os.path.basename(output_image),
      'objects': [],
      'directions': {},
  }

  # Put a plane on the ground so we can compute cardinal directions
  bpy.ops.mesh.primitive_plane_add(radius=10)
  plane = bpy.context.object

  def rand(L):
    return 2.0 * L * (random.random() - 0.5)

  # Add random jitter to camera position
  if args.camera_jitter > 0:
    for i in range(3):
      bpy.data.objects['Camera'].location[i] += rand(args.camera_jitter)
      bpy.data.objects['Camera'].location[i] *= 1.2

  # Figure out the left, up, and behind directions along the plane and record
  # them in the scene structure
  camera = bpy.data.objects['Camera']
  plane_normal = plane.data.vertices[0].normal
  cam_behind = camera.matrix_world.to_quaternion() * Vector((0, 0, -1))
  cam_left = camera.matrix_world.to_quaternion() * Vector((-1, 0, 0))
  cam_up = camera.matrix_world.to_quaternion() * Vector((0, 1, 0))
  plane_behind = (cam_behind - cam_behind.project(plane_normal)).normalized()
  plane_left = (cam_left - cam_left.project(plane_normal)).normalized()
  plane_up = cam_up.project(plane_normal).normalized()

  # Delete the plane; we only used it for normals anyway. The base scene file
  # contains the actual ground plane.
  delete_object(plane)

  mat = bpy.data.materials.new(name="Wall")
  mat.use_nodes = True

  walls = ["wall1.png", "wall2.jpg", "wall3.jpg", "wall4.jpg"]

  image_path = os.path.join("materials", random.choice(walls))

  nt = mat.node_tree
  nodes = nt.nodes
  links = nt.links

  # clear
  while(nodes): nodes.remove(nodes[0])

  output  = nodes.new("ShaderNodeOutputMaterial")
  diffuse = nodes.new("ShaderNodeBsdfDiffuse")
  texture = nodes.new("ShaderNodeTexImage")
  # uvmap   = nodes.new("ShaderNodeUVMap")
  # mapping = nodes.new("ShaderNodeMapping")

  texture.image = bpy.data.images.load(image_path)
  texture.texture_mapping.scale=(10.0, 10.0, 1.0)

  # uvmap.uv_map = "UV"

  links.new( output.inputs['Surface'], diffuse.outputs['BSDF'])
  links.new(diffuse.inputs['Color'],   texture.outputs['Color'])

  mat2 = bpy.data.materials.new(name="Floor")
  mat2.use_nodes = True

  floors = ["floor1.jpg", "floor2.png", "floor3.jpg", "floor4.jpg"]

  image_path = os.path.join("materials", random.choice(floors))

  nt = mat2.node_tree
  nodes = nt.nodes
  links = nt.links

  # clear
  while(nodes): nodes.remove(nodes[0])

  output  = nodes.new("ShaderNodeOutputMaterial")
  diffuse = nodes.new("ShaderNodeBsdfDiffuse")
  texture = nodes.new("ShaderNodeTexImage")
  # uvmap   = nodes.new("ShaderNodeUVMap")
  # mapping = nodes.new("ShaderNodeMapping")

  texture.image = bpy.data.images.load(image_path)
  texture.texture_mapping.scale=(10.0, 10.0, 1.0)

  # uvmap.uv_map = "UV"

  links.new(output.inputs['Surface'], diffuse.outputs['BSDF'])
  links.new(diffuse.inputs['Color'],   texture.outputs['Color'])

  #Check if the active object has a material slot, create one if it doesn't. 
  #Assign the material to the first slot for the active object.
  for obj in bpy.data.objects:
    if "wall" in obj.name:
      if obj.data.materials:
        obj.data.materials[0].material = mat
      else:
        obj.data.materials.append(mat)
    if "floor" in obj.name:
      if obj.data.materials:
        obj.data.materials[0].material = mat2
      else:
        obj.data.materials.append(mat2)

  # Save all six axis-aligned directions in the scene struct
  scene_struct['directions']['behind'] = tuple(plane_behind)
  scene_struct['directions']['front'] = tuple(-plane_behind)
  scene_struct['directions']['left'] = tuple(plane_left)
  scene_struct['directions']['right'] = tuple(-plane_left)
  scene_struct['directions']['above'] = tuple(plane_up)
  scene_struct['directions']['below'] = tuple(-plane_up)

  # Add random jitter to lamp positions
  if args.key_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Key'].location[i] += rand(args.key_light_jitter)
  if args.back_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Back'].location[i] += rand(args.back_light_jitter)
  if args.fill_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Fill'].location[i] += rand(args.fill_light_jitter)

  # Now make some random objects
  objects, blender_objects, urdf_objects = add_random_objects(scene_struct, num_objects, args, camera, split=split)

  # Render the scene and dump the scene data structure
  
  # while True:
  print ("rendering")

  output_node = bpy.context.scene.node_tree.nodes.new('CompositorNodeComposite')
  render_node = bpy.context.scene.node_tree.nodes['Render Layers']

  link4 = bpy.context.scene.node_tree.links.new(render_node.outputs[0], output_node.inputs[0])

  output_node2 = bpy.context.scene.node_tree.nodes.new('CompositorNodeOutputFile')
  output_node2.base_path = args.output_depth_dir
  output_node2.format.file_format = 'OPEN_EXR'
  output_node2.file_slots[0].path = output_image.split("/")[-1]
  # link5 = bpy.context.scene.node_tree.links.new(render_node.outputs[2], range_node.inputs[0])
  link6 = bpy.context.scene.node_tree.links.new(render_node.outputs[2], output_node2.inputs[0])
  bpy.ops.render.render(write_still=True)
    # break
  
  physics = True

  try:
    simulate(urdf_objects)

  except:
    physics = False

  scene_struct['physics'] = physics

  cmd = 'rm -rf %s_urdf' %args.tmp_dir
  call(cmd, shell=True)

  scene_struct['objects'] = objects
  scene_struct['relationships'] = compute_all_relationships(scene_struct)

  with open(output_scene, 'w') as f:
    json.dump(scene_struct, f, indent=2)

  if output_blendfile is not None:
    bpy.ops.wm.save_as_mainfile(filepath=output_blendfile)

def simulate(urdf_objects):
    client = p.connect(p.GUI)
    p.setGravity(0, 0, -10, physicsClientId=client) 
    urdfs = []
    from urdfpy import URDF

    for (k,urdf_object) in enumerate(urdf_objects):
      urdf = urdf_object[0]
      loc = urdf_object[1]
      ori = urdf_object[2]
      obj_name = urdf_object[3]
      if 'Cart' in obj_name:
        urdf_objects[k][1][2] -= 0.1
        loc = urdf_object[1]

      urdf_id = p.loadURDF(urdf, loc)
      urdfs.append(urdf_id)
      # numJoints = p.getNumJoints(urdf_id)
      # for j in range (numJoints):

      if (not 'Cart' in obj_name) and (not 'physics' in obj_name):
        # p.changeDynamics(urdf_id, -1, lateralFriction=1, spinningFriction=1, rollingFriction=1)
        p.changeDynamics(urdf_id, -1, mass=25)
      else:
        try:
          urdf_parse = URDF.load(urdf)
          numJoints = p.getNumJoints(urdf_id)
          # p.changeDynamics(urdf_id, -1, mass=1.0)
          for j in range (numJoints):
            joint_type = p.getJointInfo(urdf_id, j)[2] 
            if len(urdf_parse.links[j+1].visuals):
              if "wheel" in urdf_parse.links[j+1].visuals[0].name:
                p.changeDynamics(urdf_id, j, lateralFriction=0.00000000000001, spinningFriction=0.00000000000001)
                if 'Chair' in obj_name:
                  p.changeDynamics(urdf_id, j, mass=2.5)
                p.setJointMotorControl2(urdf_id,j, p.VELOCITY_CONTROL,force=1000)
              else:
                if "back" in urdf_parse.links[j+1].visuals[0].name or "seat" in urdf_parse.links[j+1].visuals[0].name:
                  p.changeDynamics(urdf_id, j, mass=10)
                if "leg" in urdf_parse.links[j+1].visuals[0].name or "central"in urdf_parse.links[j+1].visuals[0].name:
                  p.changeDynamics(urdf_id, j, mass=1)
                p.changeDynamics(urdf_id, j, lateralFriction=1, spinningFriction=1)
                p.setJointMotorControl2(urdf_id,j, p.VELOCITY_CONTROL,force=100)
                if "caster" in urdf_parse.links[j+1].visuals[0].name:
                  p.changeDynamics(urdf_id, j, mass=2.5)
                  p.setJointMotorControl2(urdf_id,j, p.VELOCITY_CONTROL,force=100)       
        except:
          print ("parse error")
          numJoints = p.getNumJoints(urdf_id)
          for j in range (numJoints):
            joint_type = p.getJointInfo(urdf_id, j)[2] 
            p.changeDynamics(urdf_id, j, mass=2.5)
            if joint_type == 0:
              p.changeDynamics(urdf_id, j, lateralFriction=0.00000000000001, spinningFriction=0.00000000000001)
              # if 'Chair' in obj_name:           
              p.setJointMotorControl2(urdf_id,j, p.VELOCITY_CONTROL,force=100)
            else:
              p.changeDynamics(urdf_id, j, lateralFriction=1, spinningFriction=1, rollingFriction=1)
              p.setJointMotorControl2(urdf_id,j, p.VELOCITY_CONTROL,force=100)
      
    import pybullet_data
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    planeId = p.loadURDF("plane.urdf", [0,0,0.02])
    p.changeDynamics(planeId, 0, lateralFriction=100000, spinningFriction=100000)
    planeId = p.loadURDF("plane.urdf", [-5,0,0], [ 0, 0.7071068, 0, 0.7071068 ])
    p.changeDynamics(planeId, 0, lateralFriction=100000, spinningFriction=100000)
    planeId = p.loadURDF("plane.urdf", [0,3,0], [ 0.7071068, 0, 0, 0.7071068  ])
    p.changeDynamics(planeId, 0, lateralFriction=100000, spinningFriction=100000)
    planeId = p.loadURDF("plane.urdf", [5,0,0], [ 0, -0.7071068, 0, 0.7071068 ])
    p.changeDynamics(planeId, 0, lateralFriction=100000, spinningFriction=100000)

    poses = []
    orns = []
    stability = []
    p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=0, cameraPitch=-30, cameraTargetPosition=[0,0,1])

    for t in range(500): 
      if t == 0:
        for u in urdfs:
          pos, orn = p.getBasePositionAndOrientation(u)
          poses.append(np.array(pos))
          orns.append(np.array(orn))

      if t == 499:
        for (idx, u) in enumerate(urdfs):
          pos, orn = p.getBasePositionAndOrientation(u)
          if np.max(abs(pos - np.array(poses[idx]))) > 0.1 or np.max(abs(orn - np.array(orns[idx]))) > 0.1:
            objects[idx]["stability"] = "no"
          else:
            objects[idx]["stability"] = "yes"

      p.stepSimulation()
      time.sleep(1/50)

    for (idx, urdf) in enumerate(urdfs):
      if objects[idx]["stability"] == "no":
        changes = []
        loc = urdf_objects[idx][1]
        new_locs = {"left": [[loc[0] - 0.5, loc[1], loc[2]], [loc[0] - 1.0, loc[1], loc[2]]], "right": [[loc[0] + 0.5, loc[1], loc[2]], [loc[0] + 1.0, loc[1], loc[2]]], "front": [[loc[0], loc[1] - 0.5, loc[2]], [loc[0], loc[1] - 1.0, loc[2]]], "behind": [[loc[0], loc[1] + 0.5, loc[2]], [loc[0], loc[1] + 1.0, loc[2]]]}
        
        for key, new_location in new_locs.items():
          for new_loc in new_location:
            for (idx2, urdf2) in enumerate(urdfs):
              loc2 = urdf_objects[idx2][1]
              p.resetBasePositionAndOrientation(urdf2, loc2, [0,0,0,1])

            poses = []
            orns = []

            p.resetBasePositionAndOrientation(urdf, new_loc, [0,0,0,1])

            for t in range(500): 

              if t == 0:
                for u in urdfs:
                  pos, orn = p.getBasePositionAndOrientation(u)
                  poses.append(np.array(pos))
                  orns.append(np.array(orn))

              if t == 499:
                for (idx3, u) in enumerate(urdfs):
                  pos, orn = p.getBasePositionAndOrientation(u)
                  if np.max(abs(pos - np.array(poses[idx]))) > 0.1 or np.max(abs(orn - np.array(orns[idx]))) > 0.1:
                    pass
                  else:
                    if idx == idx3:
                      changes.append(key)

              p.stepSimulation()
              time.sleep(1/50)

        objects[idx]["possible_change"] = [changes, new_loc]
    p.disconnect()

def add_random_objects(scene_struct, num_objects, args, camera, split):
  """
  Add random objects to the current blender scene
  """

  # Load the property file
  with open(args.properties_json, 'r') as f:
    properties = json.load(f)
    color_name_to_rgba = {}
    for name, rgb in properties['semantic']['colors'].items():
      rgba = [float(c) / 255.0 for c in rgb] + [1.0]
      color_name_to_rgba[name] = rgba
    material_mapping = [(v, k) for k, v in properties['semantic']['materials'].items()]
    semantic_list = [k for k, v in properties['semantic']['categories'].items()]
    object_list = [k for k, v in properties['object']['categories'].items()]
    weight_list = [v for k, v in properties['object']['categories'].items()]

  positions = []
  objects = []
  obj_masks = []
  obj_names = []
  blender_objects = []
  urdf_objects = []
  obj_modes = []
  normals = []

  i = 0
  tries = dict()
  from numpy.random import choice 

  invalid = 0
  while i < num_objects:
    i += 1

    #choose a mode and an object category
    modes = ["normal", "ground", "side wall"]
    weight_list2 = [0.4, 0.3, 0.3]
    mode = choice(modes, 1, p=weight_list2)[0] 

    if len(normals) > obj_modes.count("support") and (not invalid > 5):
      mode = "support"

    obj_modes.append(mode)

    if mode == "normal": invalid = 0

    if not i in tries.keys(): tries[i] = 0
    tries[i] += 1
    # Choose random categories
    
    obj_name = choice(object_list, 1, p=weight_list)[0] 

    if mode == 'support' and ('Refrigerator' in normals[-1][0] or 'Cart' in normals[-1][0] or 'Chair' in normals[-1][0]):
      obj_name = 'Chair'

    while (mode == 'normal' and obj_name == 'Cart') or (mode == "support" and obj_name == 'Bed'):
      obj_name = choice(object_list, 1, p=weight_list)[0]

    scales = {'Bed': 1.5, 'Table': 1.5, 'Refrigerator': 1.5, 'Chair': 1, 'Cart': 1.25}
    # Choose random orientation for the object.
        
    num_tries = 0
    r = scales[obj_name]
    while True:
      print ("place %d-th object"%i)
      # If we try and fail to place an object too many times, then delete all
      # the objects in the scene and start over.
      num_tries += 1
      if num_tries > args.max_retries:
        for obj in bpy.data.objects:
          if 'Table' in obj.name or 'Chair' in obj.name or 'Refrigerator' in obj.name or 'Cart' in obj.name or 'Bed' in obj.name:
            delete_object(obj)
          cmd = 'rm -rf %s'%args.tmp_dir
          call(cmd, shell=True)
        return add_random_objects(scene_struct, num_objects, args, camera)
      
      x = random.uniform(-3, 3)
      
      y = random.uniform(-8, 1)

      dists_good = True
      margins_good = True

      p_type = "static"
      if obj_name in ['Chair']:
        if random.random() > 0.25:
          p_type = "static"
        else:
          p_type = "physics"

      rot = get_rot(obj_name, mode, p_type)

      # try to place the object
      for (m,(xx, yy, rr)) in enumerate(positions):
        dx, dy = abs(x - xx), abs(y - yy)
        dist = math.sqrt(dx * dx + dy * dy)

        if abs(dx) < 0:
            print('BROKEN MARGIN!')
            margins_good = False

      if dists_good and margins_good:
        break

    theta = random.random() * 3.14

    if obj_name == 'Cart':
      base = 1.5707963

      theta = base

    if mode == "support":
      theta = normals[-1][2]

    category_path = "./data/%s.json" % obj_name.lower()
    if obj_name in ['Chair']:
      if p_type == "static":
        category_path = "./data/%s_static.json" % obj_name.lower()
      if p_type == "physics":
        category_path = "./data/%s_physics.json" % obj_name.lower()

    f = open(category_path)
    objs = json.load(f)
    if split == "val": objs = objs[:int (len(objs) * 0.14286) - 1]
    if split == "test": objs = objs[int (len(objs) * 0.14286) - 1: int (len(objs) * 0.28571) - 1]
    if split == "train": objs = objs[int (len(objs) * 0.28571) - 1:]
    if obj_name in ['Chair', 'Table', 'Bed', 'Cart']:
      id2 = random.choice(objs)
    else:
      obj = random.choice(objs)
      id2 = obj['anno_id']
 
    if obj_name == 'Cart' or p_type == "physics":
      cur_shape_dir = "%s/%s"%(args.mobility_dir, id2)
      cur_part_dir = os.path.join(cur_shape_dir, 'textured_objs')
    else:
      cur_shape_dir = "%s/%s"%(args.data_dir, id2)
      cur_part_dir = os.path.join(cur_shape_dir, 'objs')

    leaf_part_ids = [item.split('.')[0] for item in os.listdir(cur_part_dir) if item.endswith('.obj')]
    cur_render_dir = args.tmp_dir

    root_v_list = []; root_f_list = []; tot_v_num = 0;
    for idx in leaf_part_ids:
        v, f = load_obj(os.path.join(cur_part_dir, str(idx)+'.obj'))
        mesh = dict();
        mesh['v'] = v; mesh['f'] = f;
        root_v_list.append(v);
        root_f_list.append(f+tot_v_num);
        tot_v_num += v.shape[0];

    root_v = np.vstack(root_v_list)
    root_f = np.vstack(root_f_list)

    scale = np.sqrt(np.max(np.sum(root_v**2, axis=1)))
    scale /= scales[obj_name]
    root_v /= scale
    # center = np.min(root_v, axis=0) 
    # root_v -= center

    try:
      cur_result_json = os.path.join(cur_shape_dir, 'result_after_merging.json')
      with open(cur_result_json, 'r') as fin:
          tree_hier = json.load(fin)[0]
    except:
      cur_result_json = os.path.join(cur_shape_dir, 'result.json')
      with open(cur_result_json, 'r') as fin:
          tree_hier = json.load(fin)[0]
      
    obj_name2 = obj_name + str(i)
    color_dict = dict()

    part_list, part_list2, count_list, geo_list1, geo_list2 = get_list(obj_name)

    os.mkdir(os.path.join("%s_urdf", obj_name2)%args.tmp_dir) 

    part_dict = dict()
    count_dict = dict()
    objs_dict = dict()
    line_dict = dict()
    plane_dict = dict()
    final_objs = []

    _, _, part_color, part_count, part_objs, all_objects, line_geo, plane_geo = add_one_part_physics(scale, tree_hier, cur_part_dir, cur_render_dir, obj_name2, part_list, geo_list1, geo_list2, args.tmp_dir, part_dict, count_dict, objs_dict, final_objs, line_dict, plane_dict)    

    line_geo_final, plane_geo_final, part_color_all, part_color_final, part_count_final,  geometry, final_objects = revise_annotations(line_geo, plane_geo, part_color, part_count, all_objects, obj_name, part_list2, count_list, theta) 

    keep = check_part(obj_name, part_count_final, part_color_final)

    if not keep:
      i -= 1
      cmd = 'rm -rf %s'%args.tmp_dir
      call(cmd, shell=True)
      continue

    color_name, rgba = random.choice(list(color_name_to_rgba.items()))

    rendered_objs = []
    for k, v in part_objs.items():
      for val in v: 
        rendered_objs.append(val)

    part_objs['other'] = []
    part_color_all['other'] = (color_name, rgba)

    for obj_file in leaf_part_ids:
      if not obj_file in rendered_objs:
        part_objs['other'].append(obj_file)
        cur_v_list = []; cur_f_list = []; cur_v_num = 0; 
        v, f = load_obj(os.path.join(cur_part_dir, obj_file+'.obj'))
        # v -= center
        v /= scale
        cur_v_list.append(v)
        cur_f_list.append(f+cur_v_num)
        cur_v_num += v.shape[0]

        part_v = np.vstack(cur_v_list)
        part_f = np.vstack(cur_f_list)

        final_objects.append('other')
        add_mesh2 (obj_name2, 'other', part_v, part_f, args.tmp_dir, color=rgba)

    loc, ori, norm, valid = add_object2(obj_name2, (x, y), rot, normals, args.tmp_dir, theta=theta, mode=mode)
    final_location = loc
    final_orientation = ori

    if not valid:
      invalid += 1
      print ("intersecting")
      bpy.data.objects[obj_name2].select = True
      bpy.ops.object.delete() 
      cmd = 'rm -rf %s'%args.tmp_dir
      call(cmd, shell=True)
      cmd = 'rm -rf %s_urdf/%s' %(args.tmp_dir, obj_name2)
      call(cmd, shell=True)
      cmd = 'rm -rf %s_normal' %args.tmp_dir
      call(cmd, shell=True)
      obj_modes.pop(-1)
      i -= 1
      continue

    if mode == "normal":
      if len(norm):
        normals.append([obj_name2, final_objects, theta, loc, norm])
      else:
        for obj in bpy.data.objects:
          if obj_name2 in obj.name:
            obj.select = True
            bpy.ops.object.delete() 
        cmd = 'rm -rf %s' %args.tmp_dir
        call(cmd, shell=True)
        cmd = 'rm -rf %s_urdf/%s' %(args.tmp_dir, obj_name2)
        call(cmd, shell=True)
        obj_modes.pop(-1)
        i -= 1

    line_geo_final = dict()
    plane_geo_final = dict()

    geometry = True
    
    rotation_matrix = Rx(ori[0]) @ Ry(ori[1]) @ Rz(ori[2])

    for part, g in line_geo.items():
      part = rename_part(part, obj_name)
      stand = g[0]
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

    import copy
    part_color_occluded = part_color_final.copy()
    part_count_occluded = part_count_final.copy()

    ims = os.listdir(args.tmp_dir)
    images = [image for image in ims if (image.endswith(".png") and not image == "Image0001.png")]
    images.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    obj_img = np.zeros((args.height, args.width))

    assert len(images) == len(final_objects)

    keep = True

    part_masks = dict()

    for (idx, image) in enumerate(images):
      part = final_objects[idx]
      img = Image.open('%s/'%args.tmp_dir + image).convert('L')

      img = np.asarray(img)
      obj_img += img
      if len(np.where(img > 0)[0]) == 0:

        print ("occluded part: %s" %part)
        if part in part_count_occluded.keys():
          part_count_occluded[part] -= 1
          if part_count_occluded[part] == 0: 
            del part_count_occluded[part]
            if part in part_color_occluded.keys(): del part_color_occluded[part]
        else:
          if part in part_color_occluded.keys(): del part_color_occluded[part]
      else:
        if part in part_color_occluded.keys():
          # try:
          rle = binary_mask_to_rle(img)
          # compressed_rle = mask.frPyObjects(rle, rle.get('size')[0], rle.get('size')[1])
          if not part in part_masks.keys():
            part_masks[part] = []
          part_masks[part].append(rle)

    if keep:
      obj_img = np.clip(obj_img, 0, 1).astype('uint8')
      try:
        if np.min(np.where(obj_img > 0)[1]) == 0 or np.max(np.where(obj_img > 0)[1]) == 799 or np.max(np.where(obj_img > 0)[0]) == 599:
          print ("out of boundary")
          keep = False
      except:
        print ("obj image blank")
        keep = False

    if keep:
      img = Image.fromarray(obj_img*255, 'L')
      rle = binary_mask_to_rle(obj_img)
      # compressed_rle = mask.frPyObjects(rle, rle.get('size')[0], rle.get('size')[1])
      obj_mask = rle

      for (im, prev_mask) in enumerate(obj_masks):
        if mode == "support" and im == len(obj_masks) - 1: continue
        overlapping_area = prev_mask & obj_img
        img = Image.fromarray(overlapping_area*255, 'L')
        # prev1 = len(np.where(prev_mask > 0)[0])
        # print (prev1)
        # obj1 = len(np.where(obj_img > 0)[0])
        # print (obj1)
        ov = len(np.where(overlapping_area > 0)[0])
        # print (mask1 - mask2)

        if ov > 5:
          print ("overlapping objects")
          keep = False
          break

    if not keep:
      for obj in bpy.data.objects:
          if obj_name2 in obj.name:
            obj.select = True
            bpy.ops.object.delete() 
      cmd = 'rm -rf %s'%args.tmp_dir
      call(cmd, shell=True)
      cmd = 'rm -rf %s_urdf/%s' %(args.tmp_dir,obj_name2)
      call(cmd, shell=True)
      if mode == "normal": normals.pop(-1)
      obj_modes.pop(-1)
      i -= 1
      if i >= 5 and tries[i] >= 50: 
        break
      else:
        continue

    cmd = 'rm -rf %s' %args.tmp_dir 
    call(cmd, shell=True)
    
    if mode == "support":
      part_masks2 = dict()
      import copy
      part_color_occluded2 = objects[-1]["part_color"]
      part_count_occluded2 = objects[-1]["part_count"]
      final_objects = normals[-1][1]

      ims = os.listdir('%s_normal'%args.tmp_dir)
      images = [image for image in ims if (image.endswith(".png") and not image == "Image0001.png")]
      images.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

      assert len(final_objects) == len(images)

      for (idx, image) in enumerate(images):
        part = final_objects[idx]
        img = Image.open('%s_normal/'%args.tmp_dir + image).convert('L')

        img = np.asarray(img)
        obj_img += img
        if len(np.where(img > 0)[0]) == 0:
          print ("occluded part: %s" %part)
          if part in part_count_occluded2.keys():
            part_count_occluded2[part] -= 1
            if part_count_occluded2[part] == 0: 
              del part_count_occluded2[part]
              if part in part_color_occluded2.keys(): del part_color_occluded2[part]
          else:
            if part in part_color_occluded2.keys(): del part_color_occluded2[part]
        else:
          if part in part_color_occluded2.keys():
            # try:
            rle = binary_mask_to_rle(img)
            # compressed_rle = mask.frPyObjects(rle, rle.get('size')[0], rle.get('size')[1])
            if not part in part_masks2.keys():
              part_masks2[part] = []
            part_masks2[part].append(rle)

      cmd = 'rm -rf %s_normal' %args.tmp_dir 
      call(cmd, shell=True)
      
      assert part_masks != part_masks2
      objects[-1]["part_mask"] = part_masks2
      objects[-1]["part_color_occluded"] = part_color_occluded2
      objects[-1]["part_count_occluded"] = part_count_occluded2
      
    obj_masks.append(obj_img)
    obj_names.append(obj_name)
    obj = bpy.context.object
    blender_objects.append(obj)
    positions.append((x, y, r))

    urdf_file = create_urdf_object(obj_name, obj_name2, args.tmp_dir, ori, cur_shape_dir, p_type, id2)

    if p_type == "physics":
      urdf_objects.append([urdf_file, loc, ori, obj_name + "physics"])
    else:
      urdf_objects.append([urdf_file, loc, ori, obj_name])

    # # Attach a random material
    # mat_name, mat_name_out = random.choice(material_mapping)
    # add_material(mat_name, Color=rgba)

    # Record data about the object in the scene data structure
    if mode != "normal":
      pixel_coords = get_camera_coords(camera, obj.location)
    else:
      for obj in bpy.data.objects:
        if obj_name2 in obj.name:
          pixel_coords = get_camera_coords(camera, obj.location)


    part_count = dict()
    for part, count in part_count_final.items():
      if part in ['central support', 'top', 'back', 'seat']: continue
      part_count[part] = count

    part_count_occluded2 = dict()
    for part, count in part_count_occluded.items():
      if part in ['central support', 'top', 'back', 'seat']: continue
      part_count_occluded2[part] = count

    q_type = ["perception"]
    if geometry:
      q_type.append("geometry")

    objects.append({
      'category': obj_name,
      'partnet_id': id2,
      'part_count': part_count,
      'part_color': part_color_final,
      'part_count_occluded': part_count_occluded2,
      'part_color_occluded': part_color_occluded,
      'part_color_all': part_color_all,
      'part_mask': part_masks,
      'obj_mask': obj_mask,
      'original_objs': part_objs,
      '3d_coords': tuple(final_location),
      'rotation': tuple(final_orientation),
      'pixel_coords': pixel_coords,
      'scale': scale,
      'question_type': q_type
    })

  return objects, blender_objects, urdf_objects

def compute_all_relationships(scene_struct, eps=0.2):
  """
  Computes relationships between all pairs of objects in the scene.
  
  Returns a dictionary mapping string relationship names to lists of lists of
  integers, where output[rel][i] gives a list of object indices that have the
  relationship rel with object i. For example if j is in output['left'][i] then
  object j is left of object i.
  """
  all_relationships = {}
  for name, direction_vec in scene_struct['directions'].items():
    if name == 'above' or name == 'below': continue
    all_relationships[name] = []
    for i, obj1 in enumerate(scene_struct['objects']):
      coords1 = obj1['3d_coords']
      related = set()
      for j, obj2 in enumerate(scene_struct['objects']):
        if obj1 == obj2: continue
        coords2 = obj2['3d_coords']
        diff = [coords2[k] - coords1[k] for k in [0, 1, 2]]
        dot = sum(diff[k] * direction_vec[k] for k in [0, 1, 2])
        if dot > eps:
          related.add(j)
      all_relationships[name].append(sorted(list(related)))
  return all_relationships

if __name__ == '__main__':
  if INSIDE_BLENDER:
    # Run normally
    argv = extract_args()
    args = parser.parse_args(argv)
    main(args)
  elif '--help' in sys.argv or '-h' in sys.argv:
    parser.print_help()
  else:
    print('This script is intended to be called from blender like this:')
    print()
    print('blender --background --python render_images.py -- [args]')
    print()
    print('You can also run as a standalone python script to view all')
    print('arguments like this:')
    print()
    print('python render_images.py --help')

