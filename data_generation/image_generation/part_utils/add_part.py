import bpy
import os
import sys
from bpy import context

originalpath = sys.argv[6]
modelpath = sys.argv[7]
blendpath = sys.argv[8]

# bpy.ops.object.camera_add(location=(7.48, -6.5, 5.34), 
#                           rotation=(63, 0.62, 46), scale=(0.0, 0.0, 0.0)
#                           )
bpy.ops.wm.open_mainfile(filepath=originalpath)
bpy.ops.import_scene.obj(filepath = modelpath)

print('rendering')
bpy.ops.wm.save_as_mainfile(filepath=blendpath)
print('rendered')
sys.exit(0) # exit python and blender
print('exited')
