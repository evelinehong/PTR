# -*- coding: utf-8 -*-

import argparse
import random
import time
import json
import os
import sys
import numpy as np
from subprocess import call
from collections import deque

colors = {
      "gray": [87, 87, 87],
      "red": [173, 35, 35],
      "blue": [42, 75, 215],
      "green": [29, 105, 20],
      "brown": [129, 74, 25],
      "purple": [129, 38, 192],
      "cyan": [41, 208, 208],
      "yellow": [255, 238, 51]
    }

def load_obj(fn):
    fin = open(fn, 'r')
    lines = [line.rstrip() for line in fin]
    fin.close()

    vertices = []; faces = [];
    for line in lines:
        if line.startswith('v '):
            vertices.append(np.float32(line.split()[1:4]))
        elif line.startswith('f '):
            faces.append(np.int32([item.split('/')[0] for item in line.split()[1:4]]))

    f = np.vstack(faces)
    v = np.vstack(vertices)

    return v, f

def export_obj(out, v, f, color):
    color = color[:3]
    mtl_out = out.replace('.obj', '.mtl')

    with open(out, 'w') as fout:
        fout.write('mtllib %s\n' % mtl_out)
        fout.write('usemtl m1\n')
        for i in range(v.shape[0]):
            fout.write('v %f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2]))
        for i in range(f.shape[0]):
            fout.write('f %d %d %d\n' % (f[i, 0], f[i, 1], f[i, 2]))

    with open(mtl_out, 'w') as fout:
        fout.write('newmtl m1\n')
        fout.write('Kd %f %f %f\n' % (color[0], color[1], color[2]))
        fout.write('Ka 0 0 0\n')

    return mtl_out

def add_mesh(obj_name, v, f, cur_render_dir, color=[0.216, 0.494, 0.722]):
    tmp_dir = cur_render_dir
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    
    tmp_obj = os.path.join(tmp_dir, obj_name+'.obj')
    blend = tmp_obj.replace('.obj', '.blend')

    tmp_mtl = export_obj(tmp_obj, v, f, color=color)

    if obj_name+'.blend' in os.listdir(tmp_dir):
        cmd = 'bash part_utils/add_part.sh %s %s %s' % (blend, tmp_obj, blend)
    else:
        cmd = 'bash part_utils/add_part.sh part_utils/model.blend %s %s' % (tmp_obj, blend)
    call(cmd, shell=True)

    cmd = 'rm -rf %s %s %s' % (tmp_obj, tmp_mtl, blend+"1")
    call(cmd, shell=True)

def add_mesh2(obj_name, part, v, f, tmp_dir, color=[0.216, 0.494, 0.722]):
    part = part.replace(" ", "_")

    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    
    dirs = os.listdir(os.path.join("%s_urdf" %tmp_dir, obj_name))

    tmp_obj = os.path.join("%s_urdf"%tmp_dir, obj_name, obj_name + "{:03d}".format(len(dirs)) + part + ".obj")
    blend = os.path.join(tmp_dir, obj_name+'.blend')

    tmp_mtl = export_obj(tmp_obj, v, f, color=color)

    if obj_name+'.blend' in os.listdir(tmp_dir):
        cmd = 'bash part_utils/add_part.sh %s %s %s' % (blend, tmp_obj, blend)
    else:
        cmd = 'bash part_utils/add_part.sh part_utils/model.blend %s %s' % (tmp_obj, blend)

    call(cmd, shell=True)

    cmd = 'rm -rf %s' % (blend+"1")
    call(cmd, shell=True)

def add_one_part(scale, data, cur_part_dir, cur_render_dir, obj_name, part_list, geo_list1, geo_list2, part_dict = dict(), count_dict = dict(), objs_dict = dict(), final_objs = [], line_dict = dict(), plane_dict = dict()):
    
    cur_v_list = []; cur_f_list = []; cur_v_num = 0; 
    
    part = data['name']  

    part = rename_one_part(part, obj_name)

    if part in count_dict.keys():
        count_dict[part] += 1
    else:
        count_dict[part] = 1

    if 'objs' in data.keys():
        for child in data['objs']:
            v, f = load_obj(os.path.join(cur_part_dir, child+'.obj'))
            # v -= center
            v /= scale
            cur_v_list.append(v)
            cur_f_list.append(f+cur_v_num)
            cur_v_num += v.shape[0]
    if 'children' in data.keys():
        for child in data['children']:
            v, f, part_dict, count_dict, objs_dict, final_objs, line_dict, plane_dict = add_one_part(scale, child, cur_part_dir, cur_render_dir, obj_name, part_list, geo_list1, geo_list2, part_dict, count_dict, objs_dict, final_objs, line_dict, plane_dict)
            if 'objs' not in data.keys():
                cur_v_list.append(v)
                cur_f_list.append(f+cur_v_num)
                cur_v_num += v.shape[0]
    
    part_v = np.vstack(cur_v_list)
    part_f = np.vstack(cur_f_list)

    chosen_v = []

    for face in part_f:
        for fa in face:
            chosen_v.append(part_v[fa-1])

    chosen_v = np.array(chosen_v)

    if part in part_list:
        keep = True

        if part == 'chair_arm' and ('arm_near_vertical_bar' in part_dict.keys() or 'arm_horizontal_bar' in part_dict.keys()):
            keep = False
        # if part == 'chair_back' and ('back_frame_vertical_bar' in part_dict.keys() or 'back_frame_horizontal_bar' in part_dict.keys()):
        #     keep = False
        if keep:
            line_dict, plane_dict = find_equation(part, chosen_v, geo_list1, geo_list2, line_dict, plane_dict)

            final_objs.append(part)
            if not part in objs_dict.keys():
                objs_dict[part] = []
            objs_dict[part].extend(data['objs'])

            if not part in part_dict.keys():
                color_name, rgba = random.choice(list(colors.items()))
            

                if (part == 'arm_near_vertical_bar' and 'arm_horizontal_bar' in part_dict.keys()):
                    while color_name == part_dict['arm_horizontal_bar'][0]:
                        color_name, rgba = random.choice(list(colors.items()))

                if (part == 'arm_horizontal_bar' and 'arm_near_vertical_bar' in part_dict.keys()):
                    while color_name == part_dict['arm_near_vertical_bar'][0]:
                        color_name, rgba = random.choice(list(colors.items()))

                part_dict[part] = (color_name, rgba)

            color_name = part_dict[part][0]
            rgba = [float(int(c)) / 255.0 for c in part_dict[part][1]] + [1.0]
            
            add_mesh(obj_name, part_v, part_f, cur_render_dir, color=rgba)

    return part_v, part_f, part_dict, count_dict, objs_dict, final_objs, line_dict, plane_dict


def add_one_part_physics(scale, data, cur_part_dir, cur_render_dir, obj_name, part_list, geo_list1, geo_list2, tmp_dir, part_dict = dict(), count_dict = dict(), objs_dict = dict(), final_objs = [], line_dict = dict(), plane_dict = dict()):
    cur_v_list = []; cur_f_list = []; cur_v_num = 0; 
    
    part = data['name']
    part = rename_one_part(part, obj_name)

    i = 0
    if 'objs' in data.keys():
        for child in data['objs']:
            try:
                v, f = load_obj(os.path.join(cur_part_dir, child+'.obj'))
                # v -= center
                v /= scale
                cur_v_list.append(v)
                cur_f_list.append(f+cur_v_num)
                cur_v_num += v.shape[0]
                i += 1
            except:
                pass
    if 'children' in data.keys():
        for child in data['children']:
            v, f, part_dict, count_dict, objs_dict, final_objs, line_dict, plane_dict = add_one_part_physics(scale, child, cur_part_dir, cur_render_dir, obj_name, part_list, geo_list1, geo_list2, tmp_dir, part_dict, count_dict, objs_dict, final_objs, line_dict, plane_dict)
            if 'objs' not in data.keys() and (not v == np.vstack([[]])):
                cur_v_list.append(v)
                cur_f_list.append(f+cur_v_num)
                cur_v_num += v.shape[0]
                i += 1
    keep = True

    if i == 0: 
        keep = False
        cur_v_list = [[]]
        cur_f_list = [[]]
    part_v = np.vstack(cur_v_list)
    part_f = np.vstack(cur_f_list)

    
    if part == 'chair_arm' and ('arm_near_vertical_bar' in part_dict.keys() or 'arm_horizontal_bar' in part_dict.keys()):
        keep = False

    if keep:
        if part in count_dict.keys():
            count_dict[part] += 1
        else:
            count_dict[part] = 1

    chosen_v = []

    for face in part_f:
        for fa in face:
            chosen_v.append(part_v[fa-1])

    chosen_v = np.array(chosen_v)


    if part in part_list:
        if keep:
            line_dict, plane_dict = find_equation(part, chosen_v, geo_list1, geo_list2, line_dict, plane_dict)
            final_objs.append(part)
            if not part in objs_dict.keys():
                objs_dict[part] = []
            objs_dict[part].extend(data['objs'])

            if not part in part_dict.keys():
                color_name, rgba = random.choice(list(colors.items()))
                

                if (part == 'arm_near_vertical_bar' and 'arm_horizontal_bar' in part_dict.keys()):
                    while color_name == part_dict['arm_horizontal_bar'][0]:
                        color_name, rgba = random.choice(list(colors.items()))

                if (part == 'arm_horizontal_bar' and 'arm_near_vertical_bar' in part_dict.keys()):
                    while color_name == part_dict['arm_near_vertical_bar'][0]:
                        color_name, rgba = random.choice(list(colors.items()))

                part_dict[part] = (color_name, rgba)

            color_name = part_dict[part][0]
            rgba = [float(int(c)) / 255.0 for c in part_dict[part][1]] + [1.0]
            
            add_mesh2(obj_name, part, part_v, part_f, tmp_dir, color=rgba)

    return part_v, part_f, part_dict, count_dict, objs_dict, final_objs, line_dict, plane_dict

def rename_one_part(part, obj_name):
    if 'Table' in obj_name:
        if part in ['back_panel', 'vertical_side_panel', 'bottom_panel', 'vertical_divider_panel', 'vertical_front_panel']:
            part = 'frame'
        if part in ['runner', 'bar_stretcher', 'circular_stretcher', 'rocker']:
            part = 'leg bar'

    if 'Chair' in obj_name:
        if part in ['bar_stretcher', 'runner', 'rocker']:
            part = 'leg bar'

    if 'Bed' in obj_name:
        if part == 'bed_post': part = 'leg'
        # if part in ['bed_side_surface', 'surface_base']: part = 'frame'
    return part
    
def find_equation(part, chosen_v, geo_list1, geo_list2, line_dict, plane_dict):
    if part in geo_list1:
        
        angles = []
        ds = []
        j = 0

        large1 = np.argmax(chosen_v[:,0])
        large1 = chosen_v[large1]
        small1 = np.argmin(chosen_v[:,0])
        small1 = chosen_v[small1]
        large2 = np.argmax(chosen_v[:,2])
        large2 = chosen_v[large2]
        small2 = np.argmin(chosen_v[:,2])
        small2 = chosen_v[small2]
        large3 = np.argmax(chosen_v[:,1])
        large3 = chosen_v[large3]
        small3 = np.argmin(chosen_v[:,1])
        small3 = chosen_v[small3]
        sum1 = np.max(large1 - small1)
        sum2 = np.max(large2 - small2)
        sum3 = np.max(large3 - small3)

        if sum1 > sum2:
            if sum3 > sum1:
                angle = large3 - small3
            else:
                angle = large1 - small1
        else:
            if sum3 > sum2:
                angle = large3 - small3
            else:
                angle = large2 - small2               

        angle3 = np.copy(angle)
        angle /= (np.linalg.norm(angle) + 0.00001)

        for i in range (100):
            angle2 = 0
            while np.linalg.norm(angle2) < np.linalg.norm(angle3) / 2:            
                ran1 = np.random.randint(len(chosen_v[:, 0]) - 1)
                ran1 = chosen_v[ran1]
                ran2 = np.random.randint(len(chosen_v[:, 0]) - 1)
                ran2 = chosen_v[ran2]
                angle2 = ran1 - ran2

            angle2 /= (np.linalg.norm(angle2) + 0.00001)
            # d = math.min (abs(angle - angle2), abs(angle + angle2))
            distance = min(np.linalg.norm(abs(angle - angle2)), np.linalg.norm(abs(angle2 - angle)))
            if distance < 0.15:
                j += 1
            # ds.append(abs(d))

        if not part in line_dict.keys():
            line_dict[part] = []

        if j > 25 or (np.max(angle) > 0.95 and j > 20):
            line_dict[part].append(angle)
        else:
            line_dict[part].append([10000,10000,10000])
    
    if part in geo_list2:
        if not ('back' in part and ('back_frame_vertical_bar' in line_dict.keys() or 'back_frame_horizontal_bar' in line_dict.keys() or 'back_surface_vertical_bar' in line_dict.keys() or 'back_surface_horizontal_bar' in line_dict.keys())):

            angles = []
            ds = []
            i = 0
            j = 0
            while i < 100:
                ran1 = np.random.randint(len(chosen_v[:, 0]) - 1)
                ran1 = chosen_v[ran1]
                ran2 = np.random.randint(len(chosen_v[:, 0]) - 1)
                ran2 = chosen_v[ran2]
                ran3 = np.random.randint(len(chosen_v[:, 0]) - 1)
                ran3 = chosen_v[ran3]
                line1 = ran1-ran2
                line2 = ran2-ran3

                if part in ['chair_seat', 'pedestal', 'tabletop', 'shelf'] and ((abs(ran1[0] - ran2[0]) < 0.15 and abs(ran2[0] - ran3[0]) < 0.15 and abs(ran1[0] - ran3[0]) < 0.05) or (abs(ran1[2] - ran2[2]) < 0.15 and abs(ran2[2] - ran3[2]) < 0.15 and abs(ran1[2] - ran3[2]) < 0.25)):
                    j += 1
                    if j == 10000:
                        break
                    continue

                if part in ['cabinet_door_surface', 'door_frame', 'headboard',  'chair_back'] and (abs(ran1[1] - ran2[1]) < 0.15 and abs(ran2[1] - ran3[1]) < 0.15 and abs(ran1[1] - ran3[1]) < 0.25):
                    j += 1
                    if j == 10000:
                        break
                    continue
                i += 1
                    
                angle = np.cross(line1, line2)     
                if part in ['cabinet_door_surface', 'drawer_box', 'door_frame', 'headboard']:
                    angle[1] = 0.001
                    if part in ['drawer_box', 'headboard']:
                        angle[0] = 0.001
                
                angles.append(abs(angle))

            if j != 10000:
                angle = np.mean(np.array(angles), axis=0)
                

                angle /= (np.linalg.norm(angle) + 0.00001)

                for i in range (100):
                    ran1 = np.random.randint(len(chosen_v[:, 0]) - 1)
                    ran1 = chosen_v[ran1]
                    d = ran1 * angle
                    ds.append(d)
                
                ds = np.array(ds)

                if not part in plane_dict.keys():
                    plane_dict[part] = []

                if np.max(angle) > 0.95:
                    plane_dict[part].append(angle)
                else:
                    plane_dict[part].append([10000, 10000, 10000])

            else:
                if not part in plane_dict.keys():
                    plane_dict[part] = []
                plane_dict[part].append([10000, 10000, 10000])
        else:
            line_dict.pop('back_frame_vertical_bar', None)  
            line_dict.pop('back_frame_horizontal_bar', None) 
            line_dict.pop('back_surface_vertical_bar', None) 
            line_dict.pop('back_surface_horizontal_bar', None) 

    return line_dict, plane_dict

