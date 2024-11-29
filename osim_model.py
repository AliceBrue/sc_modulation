""" functions to modify OpenSim model properties"""


def modify_Millard(osim_file, ignore_tendon, fiber_damping, ignore_dyn):
    """ Modify osim Millard muscle properties"""

    file = open(osim_file, 'r')
    lines = file.readlines()
    new_lines = lines
    for l in range(len(lines)):
        line = lines[l]
        if line.split()[0].split('>')[0] == '<ignore_tendon_compliance':
            new_lines[l] = '                    <ignore_tendon_compliance>'+ignore_tendon+'</ignore_tendon_compliance>\n'
        elif line.split()[0].split('>')[0] == '<fiber_damping':
            new_lines[l] = '                    <fiber_damping>'+str(fiber_damping)+'</fiber_damping>\n'
        elif line.split()[0].split('>')[0] == '<ignore_activation_dynamics':
            new_lines[l] = '                    <ignore_activation_dynamics>'+ignore_dyn+'</ignore_activation_dynamics>\n'
    with open(osim_file, 'w') as file:
        file.writelines(new_lines)
    return osim_file


def able_Muscle(osim_file, muscles, able):
    """ Able or disable osim muscles"""

    file = open(osim_file, 'r')
    lines = file.readlines()
    new_lines = lines
    for l in range(len(lines)):
        line = lines[l]
        if len(line.split()[0].split('<')) > 1 and line.split()[0].split('<')[1] == 'Millard2012EquilibriumMuscle':
            if line.split()[1].split('"')[1] in muscles or muscles[0] == 'all':
                new_lines[l+2] = '                    <appliesForce>'+able+'</appliesForce>\n'
    with open(osim_file, 'w') as file:
        file.writelines(new_lines)
    return osim_file


def lock_Coord(osim_file, coords, locked):
    """ Lock or unlock osim coordinates"""

    file = open(osim_file, 'r')
    lines = file.readlines()
    new_lines = lines
    for l in range(len(lines)):
        line = lines[l]
        if len(line.split()[0].split('<')) > 1 and line.split()[0].split('<')[1] == 'Coordinate':
            if line.split()[1].split('"')[1] in coords:
                new_lines[l+10] = '                            <locked>'+locked+'</locked>\n'
    with open(osim_file, 'w') as file:
        file.writelines(new_lines)
    return osim_file


def modify_default_Coord(osim_file, coord, value):
    """ Modify osim coordinates default values"""

    file = open(osim_file, 'r')
    lines = file.readlines()
    new_lines = lines
    for l in range(len(lines)):
        line = lines[l]
        if len(line.split()[0].split('<')) > 1 and line.split()[0].split('<')[1] == 'Coordinate':
            if line.split()[1].split('"')[1] == coord:
                new_lines[l + 2] = '                            <default_value>'+str(value)+'</default_value>\n'
    with open(osim_file, 'w') as file:
        file.writelines(new_lines)
    return osim_file