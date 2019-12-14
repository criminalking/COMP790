import bpy
import numpy as np
import math
import mathutils
import os

"""
pipeline:
Codes:
1. Create outer eyeball
- create anterior cornea (quadric function) 
- flip normal
- set material (node editor codes, change input of IOR, the same as cornea mesh material in NVGaze), also set material of diffuse
- create eyeball (sphere)
- set material (node editor - diffuse + (optional)texture), also set material of anterior cornea
- compute intersection of two surfaces (keep respective material)
2. Create inner eyeball
- create posterior cornea (quadric function) TODO: Here we need to extend the surface!!!!!!!!
- flip normal
- set material (node editor codes, the same as anterior cornea except IOR)
- create iris (disk with a pupil hole, tab+e+s)
- set material (node editor codes - diffuse + (optional)texture)
- compute intersection of two surfaces (keep respective material)
3. Create lens mesh
- create anterior lens (quadric function)
- flip normal
- set material (node editor codes, the same as anterior cornea except IOR)
- create posterior lens (quadric function)
- set material (node editor codes, the same as anterior cornea except IOR)
- compute intersection of two surfaces (keep respective material)

Blender:
Create material node editor in blender
"""


def delete_object(name):
    # Deselect all
    bpy.ops.object.select_all(action='DESELECT')
    # Select the object
    bpy.data.objects[name].select = True
    bpy.ops.object.delete() 


def flip_normals(name):
    bpy.ops.object.select_all(action='DESELECT') # Deselect all
    bpy.data.objects[name].select = True # Select the object
    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.flip_normals() # just flip normals
    bpy.ops.object.mode_set()


def intersect_two_lenses(object1, object2):
    # compute intersection
    # boolean 1
    bool1 = object1.modifiers.new(type="BOOLEAN", name="bool 1")
    bool1.object = object2
    bool1.operation = 'INTERSECT'
    bpy.context.scene.objects.active = object1
    bpy.ops.object.modifier_apply(apply_as='DATA', modifier="bool 1")
    # boolean 2
    bool2 = object2.modifiers.new(type="BOOLEAN", name="bool 2")
    bool2.object = object1
    bool2.operation = 'INTERSECT'
    bpy.context.scene.objects.active = object2
    bpy.ops.object.modifier_apply(apply_as='DATA', modifier="bool 2")
    # delete redundant mesh
    delete_object(object2.name)


def intersect_two_surfaces(object1, object2):
    bool1 = object1.modifiers.new(type="BOOLEAN", name="bool 1")
    bool1.object = object2
    bool1.operation = 'INTERSECT'
    bpy.context.scene.objects.active = object1
    bpy.ops.object.modifier_apply(apply_as='DATA', modifier="bool 1")
    delete_object(object2.name)
    
    
def union_two_surfaces(object1, object2):
    bool1 = object1.modifiers.new(type="BOOLEAN", name="bool 1")
    bool1.object = object2
    bool1.operation = 'UNION'
    bpy.context.scene.objects.active = object1
    bpy.ops.object.modifier_apply(apply_as='DATA', modifier="bool 1")
    # delete redundant mesh
    delete_object(object2.name)
    

def create_quadric_surface(Q, R, size, div, name, offset=0.0, sign=1):
    A = 1 + Q
    B = -2 * R
    function = "sqrt((" + str(B) + ")**2/4/((" + str(A) + ")**2) - 1/" + str(A) + "*( x**2 + y**2 )) - " + str(B) + "/2/" + str(A) + " + " + str(offset)
    if  sign == -1:
        function = "-" + function
    bpy.ops.mesh.primitive_z_function_surface(equation=function, size_x=size, size_y=size, div_x=div, div_y=div)


def create_outer_mesh():
    # create outer mesh
    # create anterior cornea
    create_quadric_surface(-0.25, 7.8, size=12, div=100, name="anterior_cornea", offset=0.0, sign=-1)
    bpy.context.object.name = 'anterior_cornea'
    flip_normals("anterior_cornea")
    bpy.ops.object.material_slot_add()
    bpy.data.objects['anterior_cornea'].material_slots[:][0].material = bpy.data.materials['anterior_cornea']
    bpy.ops.object.material_slot_add()
    bpy.data.objects['anterior_cornea'].material_slots[:][1].material = bpy.data.materials['sclera']
    # create sclera(sphere)
    radius = 12
    offset_eyeball = 1 + radius
    bpy.ops.mesh.primitive_uv_sphere_add(segments=90, ring_count=45, size=radius, location=(0.0, 0.0, offset_eyeball))
    bpy.context.object.name = "sclera"
    bpy.ops.object.material_slot_add()
    bpy.data.objects['sclera'].material_slots[:][0].material = bpy.data.materials['sclera']
    bpy.ops.object.material_slot_add()
    bpy.data.objects['sclera'].material_slots[:][1].material = bpy.data.materials['anterior_cornea']
    # compute intersection of anterior cornea and outer eyeball
    anterior_cornea = bpy.data.objects['anterior_cornea'] 
    sclera = bpy.data.objects['sclera']
    union_two_surfaces(sclera, anterior_cornea)


def create_inner_mesh(accommodation, pupil_scale):
    # create inner mesh
    # create posterior cornea
    create_quadric_surface(-0.25, 6.5, size=10, div=100, name="posterior_cornea", offset=0.55, sign=-1)
    bpy.context.object.name = 'posterior_cornea'
    flip_normals("posterior_cornea")
    posterior_cornea = bpy.data.objects['posterior_cornea'] 
    bpy.ops.object.material_slot_add()
    posterior_cornea.material_slots[:][0].material = bpy.data.materials['posterior_cornea']
    bpy.ops.object.material_slot_add()
    posterior_cornea.material_slots[:][1].material = bpy.data.materials['iris']
    # create inner sclera, note this step is only for extending the posterior conrea mesh which does not have extended fucnction
    radius = 11.5
    offset_eyeball = 1 + 12 # outer and inner sclera are concentric 
    bpy.ops.mesh.primitive_uv_sphere_add(segments=90, ring_count=45, size=radius, location=(0.0, 0.0, offset_eyeball))
    bpy.context.object.name = 'inner_sclera'
    inner_sclera = bpy.data.objects['inner_sclera']
    bpy.ops.object.material_slot_add()
    inner_sclera.material_slots[:][0].material = bpy.data.materials['iris']
    bpy.ops.object.material_slot_add()
    inner_sclera.material_slots[:][1].material = bpy.data.materials['posterior_cornea']
    # compute intersection of posterior cornea and inner eyeball
    union_two_surfaces(inner_sclera, posterior_cornea)
    # create an iris cube
    # FIXME: here the z is wrong as we should modify posterior cornea mesh first
    depth = 3.52 - 0.04 * accommodation 
    bpy.ops.mesh.primitive_cylinder_add(radius=10, depth=2*depth, location=(0.0, 0.0, 0.0)) # 0.01mm before the lens
    bpy.context.object.name = "iris"
    iris = bpy.data.objects['iris']
    bpy.ops.object.material_slot_add()
    iris.material_slots[:][0].material = bpy.data.materials['iris']
    bpy.ops.object.material_slot_add()
    iris.material_slots[:][1].material = bpy.data.materials['posterior_cornea']
    # create a hole in the middle
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.object.mode_set(mode='OBJECT')
    faces = iris.data.polygons
    for face in faces:
        if face.normal[2] == 1:
            face.select = True
    bpy.ops.object.mode_set(mode = 'EDIT')
    bpy.context.tool_settings.mesh_select_mode = (False, False, True) # only select vertices
    bpy.ops.mesh.extrude_region_move(MESH_OT_extrude_region={"mirror":False})
    bpy.ops.transform.resize(value=(pupil_scale, pupil_scale, pupil_scale), constraint_axis=(False, False, False))
    bpy.ops.mesh.delete(type='FACE')
    bpy.ops.object.mode_set(mode='OBJECT')
    # intersect two surfaces
    intersect_two_surfaces(inner_sclera, iris)

    
def create_lens(accommodation):
    # create anterior lens
    create_quadric_surface(-7.52+1.29*accommodation, 12.0-0.4*accommodation, size=30, div=100, name="anterior_lens", offset=3.52-0.04*accommodation, sign=1)
    bpy.context.object.name = "anterior_lens"
    flip_normals("anterior_lens")
    bpy.ops.object.material_slot_add()
    bpy.data.objects['anterior_lens'].material_slots[:][0].material = bpy.data.materials['anterior_lens']
    bpy.ops.object.material_slot_add()
    bpy.data.objects['anterior_lens'].material_slots[:][1].material = bpy.data.materials['posterior_lens']
    # change IOR of anterior lens
    bpy.data.materials['anterior_lens'].node_tree.nodes['Value'].outputs[0].default_value = 1.42 + 0.0026 * accommodation - 0.00022 * accommodation * accommodation
    # create posterior lens, R is negative when surface is concave
    create_quadric_surface(-1.35-0.43*accommodation, -5.22+0.2*accommodation, size=30, div=100, name="posterior_lens", offset=7.29, sign=-1)
    bpy.context.object.name = "posterior_lens"
    bpy.ops.object.material_slot_add()
    bpy.data.objects['posterior_lens'].material_slots[:][0].material = bpy.data.materials['posterior_lens']
    bpy.ops.object.material_slot_add()
    bpy.data.objects['posterior_lens'].material_slots[:][1].material = bpy.data.materials['anterior_lens']
    # compute intersection
    anterior_lens = bpy.data.objects['anterior_lens'] 
    posterior_lens = bpy.data.objects['posterior_lens']
    intersect_two_lenses(anterior_lens, posterior_lens)
    bpy.context.scene.cursor_location = (0.0, 0.0, 13) # change origin of the center to this position
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects['anterior_lens'].select=True
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR')


def deg2rad(deg):
    return deg * 3.141592 / 180.0

    
def set_render_resolution(resolution):
    print('setting render resolution ', resolution)

    # disable cropping
    bpy.data.scenes["Scene"].render.border_min_y = 0.0
    bpy.data.scenes["Scene"].render.border_max_y = 1.0

    # adjust resolution
    bpy.data.scenes["Scene"].render.resolution_x = resolution[0]
    bpy.data.scenes["Scene"].render.resolution_y = resolution[1]


def createCamera(p, r, fov, name=''):
    bpy.ops.object.camera_add(view_align=True, enter_editmode=False, location=p, rotation=r, layers=(
        True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,
        False, False, False, False))

    # get cam from selection
    selected = [o for o in bpy.context.scene.objects if o.select]
    # print(selected)
    cam = selected[0]
    print('created camera ', cam.name)
    if (name != ''):
        cam.name = name

    # adjust fov
    cam.data.angle = deg2rad(fov)
    cam.data.clip_start = 0
    cam.data.draw_size = 50

    
def createSpotLight(p, intensity, fov, direction, blendRange=0.2, name=''):
    # Create new lamp datablock
    lightname = "NewSpotLight"
    if (name != ''):
        lightname = name
    lamp_data = bpy.data.lamps.new(name=lightname, type='SPOT')

    # Create new object with our lamp datablock
    lamp_object = bpy.data.objects.new(name=lightname, object_data=lamp_data)

    # Link lamp object to the scene so it'll appear in this scene
    bpy.data.scenes["Scene"].objects.link(lamp_object)

    # Place lamp to a specified location
    lamp_object.location = p
    # rotation
    lamp_object.rotation_euler = direction
    # emission angle
    lamp_object.data.spot_size = deg2rad(fov)
    # blend range
    lamp_object.data.spot_blend = blendRange

    # And finally select it make active
    # lamp_object.select = True
    bpy.data.scenes["Scene"].objects.active = lamp_object

    bpy.data.lamps[name].use_nodes = True
    bpy.data.lamps[name].node_tree.nodes["Emission"].inputs[1].default_value = intensity # default: all lights are off



def look_at(obj, lookAtPoint):
    # get camera location
    loc_obj = obj.location # matrix_world.to_translation()
    direction = loc_obj - lookAtPoint
    # # point the cameras '-Z' and use its 'X' as up
    # rot_quat = direction.to_track_quat('-Z', 'X')
    # # assume we're using euler rotation
    # obj.rotation_euler = rot_quat.to_euler()
    lengthDirection = math.sqrt(direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2])
    directionNorm = [direction[0] / lengthDirection, direction[1] / lengthDirection, direction[2] / lengthDirection]
    # get rotation about x and z (no roll about y requird)
    xrot = math.asin(directionNorm[1])
    yrot = math.asin(directionNorm[0])
    # assume we're using euler rotation XYZ
    obj.rotation_euler = (math.pi + xrot, -yrot, 0)


def set_camera_pos(camera_id, num_cameras, center, radius, lookAtPoint, shift_x=0, shift_y=0):
    camIdx = 0
    [samples, step] = np.linspace(0, 360, num_cameras, False, True)
    #angle = 90 - samples[camera_id]  # adjusting angle to match proto1
    angle = samples[camera_id] 
    p = [0, 0, 0]
    p[0] = center[0] + radius * math.cos(deg2rad(angle)) + shift_x
    p[1] = center[1] + radius * math.sin(deg2rad(angle)) + shift_y
    p[2] = center[2] 

    cam = bpy.data.objects["camera_%02d" % (camera_id)]
    cam.location = p


def generateCircularCameraArray(circleCenterPoint, circleRadius, numCameras, lookAtPoint, fov=60):

    for camera_id in range(numCameras):
        cam_name = "camera_%02d" % (camera_id)
        p = [0, 0, 0]
        r = (0, 0, 0)
        createCamera(p, r, fov, cam_name)
        set_camera_pos(camera_id, numCameras, circleCenterPoint, circleRadius, mathutils.Vector(lookAtPoint))
        cam = bpy.data.objects[cam_name]
        look_at(cam, mathutils.Vector(lookAtPoint))

    bpy.data.scenes["Scene"].camera = bpy.data.objects["camera_00"]


def generateCircularSpotLightArray(circleCenterPoint, circleRadius, numLights, lookAtPoint, intensity, fov=60, blendAreaSize=0.5):
    lightIdx = 0
    print('generateCircularSpotLightArray')

    [samples, step] = np.linspace(0, 360, numLights, False, True)
    for angle in samples:
        p = [0, 0, 0]
        p[0] = circleCenterPoint[0] + circleRadius * math.cos(deg2rad(angle))
        p[1] = circleCenterPoint[1] + circleRadius * math.sin(deg2rad(angle))
        p[2] = circleCenterPoint[2] 

        direction = [deg2rad(-90.0), 0.0, 0.0]
        lightName = "spotlight_%02d" % (lightIdx)
        createSpotLight(p, intensity, fov, direction, blendAreaSize, lightName)
        lightobject = bpy.data.objects[lightName]
        # orient spot light so that is directed towards lookat point
        look_at(lightobject, mathutils.Vector(lookAtPoint))
        lightIdx += 1

        
def change_accommodation(accommodation, pupil_scale):
    if bpy.data.objects.get("anterior_lens") is not None:
        delete_object("anterior_lens")
    if bpy.data.objects.get("inner_sclera") is not None:
        delete_object("inner_sclera")
    create_inner_mesh(accommodation, pupil_scale)
    create_lens(accommodation)


def create_eye_mesh(accommodation, pupil_scale):
    if bpy.data.objects.get("anterior_lens") is not None:
        delete_object("anterior_lens")
    if bpy.data.objects.get("inner_sclera") is not None:
        delete_object("inner_sclera")
    if bpy.data.objects.get("sclera") is not None:
        delete_object("sclera")
    create_outer_mesh()
    create_inner_mesh(accommodation, pupil_scale)
    create_lens(accommodation)

    
def rotate_eye(vert_angle, hori_angle):
    bpy.data.objects['sclera'].rotation_euler[0] = np.pi/180*vert_angle # rotate around x
    bpy.data.objects['sclera'].rotation_euler[1] = np.pi/180*hori_angle # rotate around y
    bpy.data.objects['inner_sclera'].rotation_euler[0] = np.pi/180*vert_angle # rotate around x
    bpy.data.objects['inner_sclera'].rotation_euler[1] = np.pi/180*hori_angle # rotate around y
    bpy.data.objects['anterior_lens'].rotation_euler[0] = np.pi/180*vert_angle # rotate around x
    bpy.data.objects['anterior_lens'].rotation_euler[1] = np.pi/180*hori_angle # rotate around y


def render_three_images(index, root):
    # # render image with all eye structures
    # bpy.data.scenes["Scene"].render.filepath = os.path.join(root, 'shot_%05d_all.png' % (index))
    # bpy.ops.render.render(write_still=True)
    
    # render without posterior lens
    a = bpy.data.materials['posterior_lens'].node_tree.nodes["Material Output"].inputs[0].links
    bpy.data.materials['posterior_lens'].node_tree.links.remove(a[0])
    bpy.data.scenes["Scene"].render.filepath = os.path.join(root, 'shot_%05d_noPIV.png' % (index))
    bpy.ops.render.render(write_still=True)
    bpy.data.materials['posterior_lens'].node_tree.links.new(bpy.data.materials['posterior_lens'].node_tree.nodes["Material Output"].inputs[0], bpy.data.materials['posterior_lens'].node_tree.nodes["Mix Shader Last"].outputs[0])

    # render without the whole lens
    bpy.data.objects['anterior_lens'].hide_render = True
    bpy.data.scenes["Scene"].render.filepath = os.path.join(root, 'shot_%05d_noPIIIPIV.png' % (index))
    bpy.ops.render.render(write_still=True)
    bpy.data.objects['anterior_lens'].hide_render = False


"""TODO: opencv issue in blender
def p3_in_image(index, root):
    img123 = cv2.imread(os.path.join(root, 'shot_%05d_noPIV.png' % (index)), cv2.IMREAD_GRAYSCALE)
    img12 = cv2.imread(os.path.join(root, 'shot_%05d_noPIIIPIV.png' % (index)), cv2.IMREAD_GRAYSCALE)

    # diff operation
    diff = cv2.absdiff(img123, img12)
    # get color < 20 parts
    diff = 1 - (diff > 150) * 1.0

    # opening
    kernel = np.ones((3,3), np.uint8)
    diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel)

    num_pixels = np.sum(diff < 0.5)
    if num_pixels > 20:
        return True
    else:
        return False
"""


def accommodation_experiments(num_cameras, accommodation, root):
    # simulation change of accommodation
    lightName = "spotlight_06"
    bpy.data.lamps[lightName].node_tree.nodes["Emission"].inputs[1].default_value = 100
    for i in range(num_cameras):
        bpy.data.scenes["Scene"].camera = bpy.data.objects["camera_%02d"%i]
        for j in range(len(accommodation)):
            change_accommodation(accommodation[j], pupil_scale)
            bpy.data.scenes["Scene"].render.filepath = os.path.join(root, 'shot_%02d.png' % (i*len(accommodation)+j))
            bpy.ops.render.render(write_still=True)
            #bpy.data.objects[lightName].hide_render = True


def search_position_on_eyeglasses(pupil_scale, root, quadrant, accommodation=1.0):
    # reset accommodation
    change_accommodation(accommodation, pupil_scale)

    # camera and light source are already created in model file
    cam_name = "Camera"
    cam = bpy.data.objects[cam_name]
    led_name = "Spot"
    led = bpy.data.objects[led_name]

    # render
    # led possible positions: target at eye, x: , step = 2mm
    index = 0
    lookAtPoint = (0,0,0) # eye center is in (0,0,0)

    # start render for bottom right quadrant
    # for vert_eye_angle in np.arange(-10, 0, 1): # unit: degree
    #     for hori_eye_angle in np.arange(-15, 0, 1): # unit: degree
    #         rotate_eye(vert_eye_angle, hori_eye_angle) # unit: degree
    #         for led_x in led_x_list:
    #             for led_z in led_z_list:
    #                 for led_y in np.arange(-20, 0, 2): # unit: mm
    #                     led_position = (led_x, led_y, led_z)
    #                     led.location = led_position
    #                     look_at(led, mathutils.Vector(lookAtPoint))
    #                     render_three_images(index)
    #                     index += 1

    if quadrant == 0: # top left
        led_x_min, led_x_max = -12, -2
        led_z_min, led_z_max = -20, -10
        led_y_min, led_y_max = -18, 2
        vert_angle_min, vert_angle_max = -15, 1
        hori_angle_min, hori_angle_max = 0, 16
    elif quadrant == 1: # top right
        led_x_min, led_x_max = 2, 12
        led_z_min, led_z_max = -20, -10
        led_y_min, led_y_max = -18, 2 
        vert_angle_min, vert_angle_max = -15, 1
        hori_angle_min, hori_angle_max = -15, 1
    elif quadrant == 2: # bottom left
        led_x_min, led_x_max = -12, -2
        led_z_min, led_z_max = -20, -10
        led_y_min, led_y_max = -2, 18
        vert_angle_min, vert_angle_max = 0, 16
        hori_angle_min, hori_angle_max = 0, 16
    elif quadrant == 3: # bottom right
        led_x_min, led_x_max = 2, 12
        led_z_min, led_z_max = -20, -10
        led_y_min, led_y_max = -2, 18
        vert_angle_min, vert_angle_max = 0, 16
        hori_angle_min, hori_angle_max = -15, 1

    # iterate in top right quadrant
    # iteration number: 5 x 5 x 10 x 6 x 6 x 2(piii, piv) = 18000 
    for led_x in np.arange(led_x_min, led_x_max, 2):
        for led_z in np.arange(led_z_min, led_z_max, 2): # unit: mm
            for led_y in np.arange(led_y_min, led_y_max, 2): # unit: mm
                led_position = (led_x, led_y, led_z)
                led.location = led_position
                look_at(led, mathutils.Vector(lookAtPoint))
#                num_p3 = 0
                for vert_eye_angle in np.arange(vert_angle_min, vert_angle_max, 3): # unit: degree
                    for hori_eye_angle in np.arange(hori_angle_min, hori_angle_max, 3): # unit: degree
                        rotate_eye(vert_eye_angle, hori_eye_angle) # unit: degree
                        render_three_images(index, root)
                        
                        # determine if there are 3rd purkinje image in the image
#                        if p3_in_image(index, root):
#                            num_p3 += 1
                        index += 1
                # if num_p3 > 0:
                #     # write results in the file
                #     string = "%d, %d, %d: %d\n" % (led_x, led_y, led_z, num_p3)
                #     with open("num_p3.txt", 'a') as textfile:
                #         textfile.write(string)
                #     textfile.close()
                    
        
def main():
    accommodation = 1.0
    pupil_radius = 2 # 2mm
    pupil_scale = 0.1 * pupil_radius
    num_cameras = 8
    num_lights = 8
    
    # set cursor location
    bpy.context.scene.cursor_location = (0.0, 0.0, 0.0)
    bpy.data.scenes["Scene"].render.engine = 'CYCLES'

#    # create mesh
#    create_eye_mesh(accommodation, pupil_scale)

#    # set circular cameras and leds
#    generateCircularCameraArray(circleCenterPoint=[0,0,-35], circleRadius=45, numCameras=num_cameras, lookAtPoint=[0,0,0], fov=30) # unit: mm
#    generateCircularSpotLightArray(circleCenterPoint=[0,0,-35], circleRadius=40, numLights=num_lights, lookAtPoint=[0,0,0], intensity=0, fov=24, blendAreaSize=0.5)

#    # generate one camera
#    cam_name = "camera_00"
#    createCamera(p=(0,0,0), r=(0,0,0), fov=45, name=cam_name)
#    cam = bpy.data.objects[cam_name]
#    bpy.data.scenes["Scene"].camera = bpy.data.objects[cam_name]
#
#    # generate led
#    led_name = "spotlight_00"
#    createSpotLight(p=(0,0,0), intensity=100, fov=24, direction=(0,0,0), blendRange=0.5, name=led_name)
#    led = bpy.data.objects[led_name]

    root = '/playpen/connylu/eye_simulation'
    quadrant = 0 # 0: top left, 1: top right, 2: bottom left, 3: bottom right
    search_position_on_eyeglasses(pupil_scale, root, quadrant)   

    
main()

