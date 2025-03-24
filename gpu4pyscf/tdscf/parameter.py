# Description: This file contains all the parameters used in the code
Hartree_to_eV = 27.211385050

'''
GB Radii
Ghosh, Dulal C and coworkers
The wave mechanical evaluation of the absolute radii of atoms.
Journal of Molecular Structure: THEOCHEM 865, no. 1-3 (2008): 60-67.
'''

elements_106 = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na',
'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn',
'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr',
'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb',
'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd',
'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir',
'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']

radii = [0.5292, 0.3113, 1.6283, 1.0855, 0.8141, 0.6513, 0.5428, 0.4652, 0.4071, 0.3618,
2.165, 1.6711, 1.3608, 1.1477, 0.9922, 0.8739, 0.7808, 0.7056, 3.293, 2.5419,
2.4149, 2.2998, 2.1953, 2.1, 2.0124, 1.9319, 1.8575, 1.7888, 1.725, 1.6654,
1.4489, 1.2823, 1.145, 1.0424, 0.9532, 0.8782, 3.8487, 2.9709, 2.8224, 2.688,
2.5658, 2.4543, 2.352, 2.2579, 2.1711, 2.0907, 2.016, 1.9465, 1.6934, 1.4986,
1.344, 1.2183, 1.1141, 1.0263, 4.2433, 3.2753, 2.6673, 2.2494, 1.9447, 1.7129,
1.5303, 1.383, 1.2615, 1.1596, 1.073, 0.9984, 0.9335, 0.8765, 0.8261, 0.7812,
0.7409, 0.7056, 0.6716, 0.6416, 0.6141, 0.589, 0.5657, 0.5443, 0.5244, 0.506,
1.867, 1.6523, 1.4818, 1.3431, 1.2283, 1.1315, 4.4479, 3.4332, 3.2615, 3.1061,
2.2756, 1.9767, 1.7473, 1.4496, 1.2915, 1.296, 1.1247, 1.0465, 0.9785, 0.9188,
0.8659, 0.8188, 0.8086]
exp = [1/(i*1.8897259885789)**2 for i in radii]

ris_exp = dict(zip(elements_106,exp))

'''
range-separated hybrid functionals, (omega, alpha, beta)
'''
rsh_func = {}
rsh_func['wb97'] = (0.4, 0, 1.0)
rsh_func['wb97x'] = (0.3, 0.157706, 0.842294)  # wb97 family, a+b=100% Long-range HF exchange
rsh_func['wb97x-d'] = (0.2, 0.22, 0.78)
rsh_func['wb97x-d3'] = (0.25, 0.195728, 0.804272)
rsh_func['wb97x-v'] = (0.30, 0.167, 0.833)
rsh_func['wb97x-d3bj'] = (0.30, 0.167, 0.833)
rsh_func['cam-b3lyp'] = (0.33, 0.19, 0.46) # a+b=65% Long-range HF exchange
rsh_func['lc-blyp'] = (0.33, 0, 1.0)
rsh_func['lc-PBE'] = (0.47, 0, 1.0)

'''
pure or hybrid functionals, hybrid component a_x
'''
hbd_func = {}
hbd_func['pbe'] = 0
hbd_func['pbe,pbe'] = 0
hbd_func['tpss'] = 0
hbd_func['tpssh'] = 0.1
hbd_func['b3lyp'] = 0.2
hbd_func['pbe0'] = 0.25
hbd_func['bhh-lyp'] = 0.5
hbd_func['m05-2x'] = 0.56
hbd_func['m06'] = 0.27
hbd_func['m06-2x'] = 0.54
hbd_func['hf'] = 1
hbd_func[None] = 1




''' for sTDA
    a dictionary of chemical hardness, by mappig two lists:
   list of elements 1-94
   list of hardness for elements 1-94, floats, in Hartree
'''
elements = ['H' , 'He', 'Li', 'Be', 'B' , 'C' , 'N' , 'O' , 'F' , 'Ne',
'Na', 'Mg', 'Al', 'Si', 'P' , 'S' , 'Cl', 'Ar', 'K' , 'Ca','Sc', 'Ti',
'V' , 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn','Ga', 'Ge', 'As', 'Se',
'Br', 'Kr', 'Rb', 'Sr', 'Y' , 'Zr','Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd',
'Ag', 'Cd', 'In', 'Sn','Sb', 'Te', 'I' , 'Xe', 'Cs', 'Ba', 'La', 'Ce',
'Pr', 'Nd','Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
'Lu', 'Hf', 'Ta', 'W' , 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg','Tl', 'Pb',
'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U' , 'Np', 'Pu']
hardness = [0.47259288,0.92203391,0.17452888,0.25700733,0.33949086,
0.42195412,0.50438193,0.58691863,0.66931351,0.75191607,0.17964105,
0.22157276,0.26348578,0.30539645,0.34734014,0.38924725,0.43115670,
0.47308269,0.17105469,0.20276244,0.21007322,0.21739647,0.22471039,
0.23201501,0.23933969,0.24665638,0.25398255,0.26128863,0.26859476,
0.27592565,0.30762999,0.33931580,0.37235985,0.40273549,0.43445776,
0.46611708,0.15585079,0.18649324,0.19356210,0.20063311,0.20770522,
0.21477254,0.22184614,0.22891872,0.23598621,0.24305612,0.25013018,
0.25719937,0.28784780,0.31848673,0.34912431,0.37976593,0.41040808,
0.44105777,0.05019332,0.06762570,0.08504445,0.10247736,0.11991105,
0.13732772,0.15476297,0.17218265,0.18961288,0.20704760,0.22446752,
0.24189645,0.25932503,0.27676094,0.29418231,0.31159587,0.32902274,
0.34592298,0.36388048,0.38130586,0.39877476,0.41614298,0.43364510,
0.45104014,0.46848986,0.48584550,0.12526730,0.14268677,0.16011615,
0.17755889,0.19497557,0.21240778,0.07263525,0.09422158,0.09920295,
0.10418621,0.14235633,0.16394294,0.18551941,0.22370139]
HARDNESS = dict(zip(elements,hardness))

def gen_sTDA_alpha_beta_ax(a_x):

    ''' NA is for Hartree-Fork

        RSH functionals have specific a_x, beta, alpha values;
        hybride fucntionals have fixed alpha12 and beta12 values,
        with different a_x values, by which create beta, alpha
    '''
    beta1 = 0.2
    beta2 = 1.83
    alpha1 = 1.42
    alpha2 = 0.48

    beta = beta1 + beta2 * a_x
    alpha = alpha1 + alpha2 * a_x

    return alpha, beta